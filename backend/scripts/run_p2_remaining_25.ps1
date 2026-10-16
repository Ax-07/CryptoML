#requires -Version 5.1
<#
Historical P2 research only: 25 untouched individual feature removals.
No P3 activation, no ADA, no fold OOS and no final 90d scoring.
Smoke: 1 fold * 2 horizons * (baseline + 25 removals) = 52 fits.
Full: 36 folds * 2 horizons * (baseline + 25 removals) = 1872 fits.
Combines the new results with the preexisting five candle-feature removals.
#>
[CmdletBinding()]
param(
    [string]$Python = "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe",
    [string]$DataRoot = "E:\cryptoML\data",
    [string]$OutputRoot = "E:\cryptoML-walkforward-results",
    [switch]$SmokeOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Remaining = @(
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
    "ema_gap_12", "ema_gap_48", "rsi_14",
    "volatility_12", "volatility_48", "atr_pct_14",
    "volume_relative_12", "volume_relative_48", "quote_volume_relative_48",
    "trade_count_relative_48", "taker_buy_share",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "btc_ret_3", "btc_volatility_12", "eth_ret_3", "eth_volatility_12"
)
$AlreadyDone = @(
    "body_relative", "upper_wick_relative", "lower_wick_relative",
    "close_location", "range_pct"
)
if ($Remaining.Count -ne 25 -or $AlreadyDone.Count -ne 5) {
    throw "Unexpected predefined feature counts."
}
if (@(Compare-Object -ReferenceObject $Remaining -DifferenceObject $AlreadyDone -IncludeEqual |
      Where-Object SideIndicator -eq "==").Count -ne 0) {
    throw "Already-tested candle features must not be repeated."
}

$Smoke = Join-Path $OutputRoot "p2_feature_reduction_individual_remaining25_smoke_v1.json"
$Full = Join-Path $OutputRoot "p2_feature_reduction_individual_remaining25_full_v1.json"
$Prior = Join-Path $OutputRoot "p2_candle_features_individual_full_v1.json"
$SummaryCsv = Join-Path $OutputRoot "p2_feature_reduction_all30_individual_summary_v1.csv"
$Backend = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "Historical source folder not found: $DataRoot"
}
if (-not (Test-Path -LiteralPath $OutputRoot -PathType Container)) {
    throw "Results folder not found: $OutputRoot"
}

function Read-CheckedReport {
    param(
        [string]$Path,
        [int]$ExpectedFolds,
        [int]$ExpectedFits,
        [string[]]$ExpectedFeatures
    )
    $p = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($p.mode -ne "individual" -or
        $p.folds_analyzed -ne $ExpectedFolds -or
        $p.expected_fit_count -ne $ExpectedFits -or
        $p.feature_count -ne 30 -or
        $p.ADA_loaded_or_scored -ne $false -or
        $p.fold_OOS_scored -ne $false -or
        $p.final_90_days_scored -ne $false -or
        $p.frozen_P3_models_modified -ne $false) {
        throw "Report fails protocol checks: $Path"
    }
    $expected = @($ExpectedFeatures | ForEach-Object { "drop_$_" } | Sort-Object)
    $observed = @($p.removed_features.PSObject.Properties |
                  Where-Object { $_.Name -ne "baseline_30" } |
                  ForEach-Object { $_.Name } | Sort-Object)
    if (@(Compare-Object -ReferenceObject $expected -DifferenceObject $observed).Count -ne 0) {
        throw "Report includes an incorrect feature subset: $Path"
    }
    return $p
}

function Invoke-Ablation {
    param([string]$ReportPath, [switch]$OneFold)
    if (Test-Path -LiteralPath $ReportPath) {
        Write-Host "Report exists; refusing overwrite: $ReportPath"
        return
    }
    $cmdArgs = @(
        "-u", "-m", "mlpipeline.feature_reduction_diagnostics",
        "--mode", "individual", "--only-features"
    ) + $Remaining + @(
        "--data-root", $DataRoot,
        "--output", $ReportPath,
        "--months", "24",
        "--n-estimators", "150",
        "--seed", "42",
        "--verify-hashes"
    )
    if ($OneFold) { $cmdArgs += @("--max-folds", "1") }
    Write-Host "Starting feature-ablation run: $ReportPath"
    & $Python @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Ablation Python failed (exit code $LASTEXITCODE). No subsequent stage will run."
    }
    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        throw "No JSON report produced: $ReportPath"
    }
}

Push-Location $Backend
try {
    Invoke-Ablation -ReportPath $Smoke -OneFold
    $null = Read-CheckedReport -Path $Smoke -ExpectedFolds 1 -ExpectedFits 52 -ExpectedFeatures $Remaining
    Write-Host "Smoke validated: 52 fits, 1 fold." -ForegroundColor Green
    if ($SmokeOnly) { return }

    Invoke-Ablation -ReportPath $Full
    $current = Read-CheckedReport -Path $Full -ExpectedFolds 36 -ExpectedFits 1872 -ExpectedFeatures $Remaining
    Write-Host "Full 25-feature ablation validated: 36 folds, 1872 expected fits." -ForegroundColor Green

    if (-not (Test-Path -LiteralPath $Prior -PathType Leaf)) {
        Write-Warning "Previous 5-feature report not present; full 25-feature report was saved."
        return
    }
    $previous = Read-CheckedReport -Path $Prior -ExpectedFolds 36 -ExpectedFits 432 -ExpectedFeatures $AlreadyDone
    if ($previous.dataset_fingerprint_sha256 -ne $current.dataset_fingerprint_sha256 -or
        $previous.feature_schema -ne $current.feature_schema) {
        throw "Cannot merge old and new reports: dataset fingerprint or schema changed."
    }

    $rows = foreach ($h in @("H2", "H6")) {
        foreach ($report in @($previous, $current)) {
            foreach ($entry in $report.summary.$h.paired_drop_vs_full.PSObject.Properties) {
                $v = $entry.Value
                [PSCustomObject]@{
                    Horizon = $h
                    Feature = $entry.Name -replace "^drop_", ""
                    GainUplift = [math]::Round($v.mean_uplift_delta_points, 3)
                    FoldsPositifs = $v.positive_uplift_folds
                    FoldsComparables = $v.paired_feasible_folds
                    DeltaCouverture = [math]::Round($v.mean_realized_coverage_delta_points, 3)
                    GainLogLoss = [math]::Round($v.mean_log_loss_improvement, 6)
                }
            }
        }
    }
    if (@($rows).Count -ne 60) {
        throw "Combined summary must contain exactly 60 H2/H6 entries."
    }
    $sorted = @($rows | Sort-Object -Property Horizon, @{Expression = "GainUplift"; Descending = $true})
    if (Test-Path -LiteralPath $SummaryCsv) {
        Write-Warning "Prior 60-row summary exists; it will not be overwritten: $SummaryCsv"
    } else {
        $sorted | Export-Csv -LiteralPath $SummaryCsv -Encoding UTF8 -NoTypeInformation
        Write-Host "Saved 60-row summary: $SummaryCsv"
    }
    $sorted | Format-Table -AutoSize
    Write-Host "Exploratory inspected dev checks only. No automatic feature selection."
}
finally {
    Pop-Location
}
