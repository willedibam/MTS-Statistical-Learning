# ============================================================================
# OVERNIGHT RUN SCRIPT - dev+++ Profile (M=25-35, T=1500-2500)
# ============================================================================
# Purpose: Run all 15 generators with optimized M/T settings for overnight execution
# Expected runtime: 8-12 hours total (~30-60 min per model)
# Profile: dev+++ (prioritizes M for SPI-SPI correlation power, sufficient T)
#
# CRITICAL: Configure Windows power settings BEFORE running (see POWER_SETTINGS.md)
# ============================================================================

# Safety configuration
$ErrorActionPreference = "Continue"  # Don't stop entire run if one model fails
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = "overnight_run_${timestamp}.log"

# Activate virtual environment if it exists
$venvPath = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venvPath
} else {
    Write-Host "WARNING: No .venv found. Using system Python." -ForegroundColor Yellow
}

# Runtime tracking
$startTime = Get-Date
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "OVERNIGHT RUN STARTED: $startTime" -ForegroundColor Cyan
Write-Host "Profile: dev+++" -ForegroundColor Cyan
Write-Host "Expected runtime: 8-12 hours" -ForegroundColor Cyan
Write-Host "Log file: $logFile" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Log function
function Write-Log {
    param([string]$Message)
    $entry = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $entry
    Add-Content -Path $logFile -Value $entry
}

# ============================================================================
# PRE-RUN CHECKS
# ============================================================================

Write-Log "Starting pre-run checks..."

# Check Python environment
Write-Log "Checking Python environment..."
$pythonCheck = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Python not found. Aborting."
    exit 1
}
Write-Log "Python version: $pythonCheck"

# Check pyspi installation
Write-Log "Checking pyspi installation..."
try {
    python -c "import pyspi" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Log "pyspi is installed"
    } else {
        throw "pyspi import failed"
    }
} catch {
    Write-Log "ERROR: pyspi not installed. Run 'pip install pyspi' first. Aborting."
    exit 1
}

# Check power settings (CRITICAL)
Write-Log "Checking power settings (AC standby timeout)..."
$powerCheck = powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String -Pattern "Current AC Power Setting Index:"
if ($powerCheck -match "0x00000000") {
    Write-Log "[OK] Power settings OK: AC standby disabled (never sleep)"
} else {
    Write-Log "[WARNING] Computer may sleep during run. See POWER_SETTINGS.md to configure."
    Write-Log "   Current AC standby: $powerCheck"
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/N)"
    if ($response -ne "y") {
        Write-Log "Run aborted by user."
        exit 0
    }
}

# Check disk space
Write-Log "Checking disk space..."
$drive = (Get-Location).Drive.Name
$freeSpace = (Get-PSDrive $drive).Free / 1GB
Write-Log "Free space on ${drive}: $([math]::Round($freeSpace, 2)) GB"
if ($freeSpace -lt 10) {
    Write-Log "[WARNING] Low disk space (<10 GB). Results may be large."
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/N)"
    if ($response -ne "y") {
        Write-Log "Run aborted by user."
        exit 0
    }
}

Write-Log "Pre-run checks complete."
Write-Log ""

# ============================================================================
# MAIN RUN - ALL 15 GENERATORS
# ============================================================================

Write-Log "Starting main run with dev+++ profile..."
Write-Log ""

# List of all 15 generators (order: baseline, oscillatory, chaotic, nonlinear, directionality)
$generators = @(
    "VAR(1)",
    "OU-network",
    "Kuramoto",
    "Stuart-Landau",
    "Lorenz-96",
    "Rössler-coupled",
    "OU-heavyTail",
    "GBM-returns",
    "TimeWarp-clones",
    "CML-logistic",
    "Cauchy-OU",
    "Unidirectional-Cascade",
    "Quadratic-Coupling",
    "Exponential-Transform",
    "Phase-Lagged-Oscillators"
)

# Results tracking
$successCount = 0
$failureCount = 0
$failedModels = @()

# Run each generator
foreach ($gen in $generators) {
    $modelStartTime = Get-Date
    Write-Log "================================================"
    Write-Log "Running generator: $gen"
    Write-Log "Start time: $modelStartTime"
    Write-Log "Expected runtime: 30-60 min"
    Write-Log "================================================"
    
    # Run computation
    try {
        python -m spimts compute --mode dev+++ --subset pilot0 --models "$gen" --outdir results --include-noise --cache cache --normalise 1 --skip-existing 2>&1 | Tee-Object -FilePath $logFile -Append
        
        if ($LASTEXITCODE -eq 0) {
            $modelEndTime = Get-Date
            $modelDuration = ($modelEndTime - $modelStartTime).TotalMinutes
            Write-Log "[SUCCESS] $gen completed in $([math]::Round($modelDuration, 2)) minutes"
            $successCount++
        } else {
            throw "Python exited with code $LASTEXITCODE"
        }
    } catch {
        $modelEndTime = Get-Date
        $modelDuration = ($modelEndTime - $modelStartTime).TotalMinutes
        Write-Log "[FAILED] $gen failed after $([math]::Round($modelDuration, 2)) minutes"
        Write-Log "Error: $_"
        $failureCount++
        $failedModels += $gen
    }
    
    Write-Log ""
}

# ============================================================================
# POST-RUN SUMMARY
# ============================================================================

$endTime = Get-Date
$totalDuration = ($endTime - $startTime).TotalHours

Write-Log "================================================"
Write-Log "OVERNIGHT RUN COMPLETED"
Write-Log "================================================"
Write-Log "Start time: $startTime"
Write-Log "End time: $endTime"
Write-Log "Total runtime: $([math]::Round($totalDuration, 2)) hours"
Write-Log ""
Write-Log "Results:"
Write-Log "  [SUCCESS] Successful: $successCount / $($generators.Count)"
Write-Log "  [FAILED] Failed: $failureCount / $($generators.Count)"

if ($failureCount -gt 0) {
    Write-Log ""
    Write-Log "Failed models:"
    foreach ($model in $failedModels) {
        Write-Log "  - $model"
    }
}

Write-Log ""
Write-Log "Output directory: results/dev+++"
Write-Log "Log file: $logFile"
Write-Log "================================================"

# Final summary to screen
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "OVERNIGHT RUN SUMMARY" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Total runtime: $([math]::Round($totalDuration, 2)) hours" -ForegroundColor Green
Write-Host "Successful: $successCount / $($generators.Count)" -ForegroundColor Green
Write-Host "Failed: $failureCount / $($generators.Count)" -ForegroundColor $(if ($failureCount -gt 0) { "Red" } else { "Green" })
Write-Host "Log file: $logFile" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Exit code
if ($failureCount -gt 0) {
    exit 1
} else {
    exit 0
}
