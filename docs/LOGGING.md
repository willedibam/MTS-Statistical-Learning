# Logging System Documentation

**Last Updated:** October 25, 2025  
**Module:** `spimts/helpers/logging_config.py`

---

## Overview

The `spimts` package uses Python's built-in `logging` module with a **hybrid dual-output architecture**:
- **Terminal output**: Always visible, color-coded, real-time progress
- **File output**: Optional, controlled by `--log-file` argument

This design provides:
✅ Real-time visibility during runs (no PowerShell buffering)  
✅ User control over when/where logs are saved  
✅ No special character encoding issues  
✅ Structured, parseable log format with timestamps  

---

## Usage

### Basic Usage (Terminal Only)

```powershell
# No logging to file - just terminal output
python -m spimts visualize --profile dev++ --all-runs

# Output appears immediately in color
INFO     Found 16 model run(s) to visualize
INFO     Profile: dev++ | MPI heatmaps: core
INFO     
INFO     [1/16] Kuramoto (run_id=e0eae877)
INFO       Loaded 19 SPIs from cached artifacts
INFO       Saved MTS heatmap (T=5000, M=10)
...
```

### Auto-Named Log File

```powershell
# Automatically creates ./logs/visualize_dev++_YYYYMMDD-HHMMSS.log
python -m spimts visualize --profile dev++ --all-runs --log-file auto
```

**Output:**
- Terminal: Colored, real-time progress (same as above)
- File: `./logs/visualize_dev++_20251025-190530.log` with timestamps

**File format:**
```
[19:05:30] INFO     Found 16 model run(s) to visualize
[19:05:30] INFO     Profile: dev++ | MPI heatmaps: core
[19:05:30] INFO     
[19:05:31] INFO     [1/16] Kuramoto (run_id=e0eae877)
[19:05:31] INFO       Loaded 19 SPIs from cached artifacts
[19:05:32] INFO       Saved MTS heatmap (T=5000, M=10)
...
```

### Custom Log File Path

```powershell
# Save to specific location
python -m spimts visualize --profile dev++ --all-runs --log-file "C:\MyLogs\viz_run1.log"

# Or relative path
python -m spimts visualize --profile dev++ --all-runs --log-file ".\outputs\test.log"
```

---

## How It Works

### 1. Dual Handler Architecture

The logging system uses **two handlers**:

**StreamHandler (always active)**:
- Outputs to `stdout` (terminal)
- Level: `INFO` and above
- Format: `%(levelname)-8s %(message)s` (no timestamps in terminal)
- Color-coded by level (INFO=green, WARNING=yellow, ERROR=red)

**FileHandler (optional)**:
- Outputs to file path specified by `--log-file`
- Level: `DEBUG` and above (file gets more detail than terminal)
- Format: `[%(asctime)s] %(levelname)-8s %(message)s`
- No color codes (clean text file)

### 2. Color Coding (Terminal Only)

Colors use ANSI escape codes (compatible with Windows 10+ PowerShell):
- **DEBUG**: Cyan
- **INFO**: Green
- **WARNING**: Yellow
- **ERROR**: Red
- **CRITICAL**: Magenta

Colors are **automatically disabled** in file output.

### 3. Log Levels

| Level | When to Use | Appears In |
|-------|-------------|------------|
| `DEBUG` | Internal diagnostics, verbose details | File only |
| `INFO` | Progress updates, normal operations | Terminal + File |
| `WARNING` | Non-fatal issues (e.g., NaN values, skipped plots) | Terminal + File |
| `ERROR` | Fatal errors that stop processing | Terminal + File |
| `CRITICAL` | Catastrophic failures | Terminal + File |

**Current usage:**
- `logger.info()`: Model progress, plot counts, completion messages
- `logger.warning()`: Missing data, skipped visualizations
- `logger.error()`: File not found, invalid arguments

---

## Integration in Scripts

### visualize.py

```python
from spimts.helpers.logging_config import setup_logging

def main():
    p = argparse.ArgumentParser(...)
    p.add_argument("--log-file", default=None, 
                   help="Log to file: 'auto' (auto-named), explicit path, or omit")
    args = p.parse_args()
    
    # Setup logging (must be called BEFORE any logger.info() calls)
    logger = setup_logging('visualize', args.profile, args.log_file)
    
    logger.info(f"Found {len(runs)} model run(s) to visualize")
    logger.info(f"Profile: {args.profile} | MPI heatmaps: {args.include_mpi_heatmaps}")
    
    for idx, (rid, model_safe, model_dir, _mtime) in enumerate(runs, 1):
        logger.info(f"[{idx}/{len(runs)}] {model_safe} (run_id={rid})")
        # ... process model ...
        logger.info(f"  Loaded {len(all_spi_names)} SPIs from cached artifacts")
        logger.info(f"  Saved MTS heatmap (T={X.shape[0]}, M={X.shape[1]})")
        logger.info(f"  Saved {len(mpi_spis_to_plot)} MPI heatmaps")
        logger.info(f"  Generating SPI-space plots (grid + individual)...")
        logger.info(f"  Saved 3 grid plots + {n} individual plots per method")
    
    logger.info("All visualizations complete!")
```

### compute.py (to be implemented similarly)

```python
from spimts.helpers.logging_config import setup_logging

def main():
    ap = argparse.ArgumentParser(...)
    ap.add_argument("--log-file", default=None, ...)
    args = ap.parse_args()
    
    logger = setup_logging('compute', args.mode, args.log_file)
    
    logger.info(f"COMPUTE MODE: {args.mode} | Models: {total_models}")
    
    for idx, (model, gen) in enumerate(generators.items(), 1):
        logger.info(f"[{idx}/{total_models}] {model} M={M} T={T}")
        # ... compute SPIs ...
        logger.info(f"  [OK] {len(spi_names)} SPIs | {elapsed:.1f}s | ETA: {eta:.0f}s")
    
    logger.info(f"COMPLETED: {len(perf_data)}/{total_models} models in {total_elapsed:.1f}s")
```

---

## Design Decisions

### Why NOT automatic logging?

**Decision**: Logging only happens when user explicitly provides `--log-file`

**Reasoning**:
1. **User control**: Quick test runs don't need logs cluttering `./logs/`
2. **Flexibility**: User might want logs in different locations (network drive, temp folder)
3. **Separation of concerns**: Script does work, user decides if/where to log
4. **No surprises**: User knows exactly when/where files are created

### Why dual output?

**Decision**: Terminal output happens regardless of file logging

**Reasoning**:
1. **Always visible**: User can see progress even without `--log-file`
2. **Real-time feedback**: No PowerShell buffering delays
3. **File is optional backup**: Terminal is primary interface, file is archival

### Why color in terminal but not file?

**Decision**: `ColoredFormatter` strips ANSI codes from file output

**Reasoning**:
1. **Terminal readability**: Colors help scan log output quickly
2. **File parsability**: Plain text easier to grep/parse/diff
3. **Cross-platform**: ANSI codes don't render in all text editors

### Why timestamps only in file?

**Decision**: Terminal shows level + message, file shows time + level + message

**Reasoning**:
1. **Terminal brevity**: User watching live doesn't need exact timestamps
2. **File context**: Timestamps critical for post-hoc debugging ("When did this fail?")
3. **Cleaner terminal**: Less visual clutter during interactive use

---

## Troubleshooting

### "Colors not showing in terminal"

**Cause**: ANSI color codes not supported by your terminal

**Fix**: Use Windows Terminal or PowerShell 7+ (not legacy cmd.exe)

### "Log file not created"

**Possible causes**:
1. Forgot `--log-file` argument → **Expected behavior** (no file = terminal only)
2. Path doesn't exist and parent directory not creatable → Check permissions
3. File path has invalid characters → Use valid filename

**Check**: Look for error message from logging setup

### "Terminal output appears all at once at the end"

**Cause**: Python's stdout buffering or PowerShell redirect (`2>&1 |`)

**Fix 1**: Run directly without redirect:
```powershell
python -m spimts visualize --profile dev++ --log-file auto
```

**Fix 2**: If you MUST use PowerShell redirect, add explicit flush:
```python
logger.info("message")
sys.stdout.flush()  # Force immediate output
```

**Fix 3**: Use `python -u` (unbuffered mode):
```powershell
python -u -m spimts visualize --profile dev++ --all-runs
```

### "RuntimeWarning clutter in logs"

**Cause**: NumPy warnings from third-party libraries (scipy, sklearn) appear on stderr

**These are NOT from your logger** - they bypass the logging system entirely.

**Fix 1**: Suppress warnings globally (not recommended):
```python
import warnings
warnings.filterwarnings("ignore")
```

**Fix 2**: Suppress specific warnings:
```python
import numpy as np
np.seterr(invalid='ignore')  # Ignore "invalid value encountered" warnings
```

**Fix 3**: Accept them (recommended) - they indicate NaN/Inf data was handled correctly

---

## Advanced: Progress Bars

The logging system includes a helper function for ASCII progress bars:

```python
from spimts.helpers.logging_config import log_progress_bar

for i, item in enumerate(items, 1):
    log_progress_bar(logger, i, len(items), prefix=f"Processing {item}")
```

**Output:**
```
INFO     Processing model1 [=====>            ] 5/20 (25%)
INFO     Processing model2 [=========>        ] 10/20 (50%)
INFO     Processing model3 [===============>  ] 18/20 (90%)
```

**Note**: Uses only ASCII characters (no special Unicode blocks) for PowerShell compatibility.

---

## Migration Guide (Old vs New)

### OLD (before logging system):

```python
print(f"[VIZ] {model_safe} (run_id={rid})")
print(f"[INFO] Loaded {len(spis)} SPIs")
print(f"[WARN] Skipping invalid data")
print(f"[ERR] File not found: {path}")
```

**Problems:**
- No timestamps
- No color
- No file output (unless user manually redirects)
- `print()` doesn't flush immediately
- Mix of `[INFO]`, `[WARN]`, `[ERR]` prefixes (inconsistent)

### NEW (with logging system):

```python
logger = setup_logging('visualize', profile, log_file)

logger.info(f"{model_safe} (run_id={rid})")
logger.info(f"Loaded {len(spis)} SPIs")
logger.warning(f"Skipping invalid data")
logger.error(f"File not found: {path}")
```

**Benefits:**
- Automatic timestamps (in file)
- Color-coded (in terminal)
- Dual output (terminal + optional file)
- Immediate flush (real-time visibility)
- Consistent format

---

## Summary

**Terminal-only workflow (quick tests):**
```powershell
python -m spimts visualize --profile dev++ --models "Kuramoto"
# See immediate colored output, no files created
```

**Full run with logging (overnight jobs):**
```powershell
python -m spimts visualize --profile dev++ --all-runs --log-file auto
# Terminal shows progress in color
# File saved to ./logs/visualize_dev++_20251025-190530.log
```

**Why this design?**
- **User control**: Logging only when you want it
- **Real-time visibility**: No PowerShell buffering delays
- **Clean terminal**: Colors + concise format for interactive use
- **Archival file**: Timestamps + full detail for post-hoc analysis
- **No surprises**: Explicit opt-in, predictable behavior
