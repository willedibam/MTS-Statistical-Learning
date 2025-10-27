# Logging Implementation Summary

**Date:** October 25, 2025  
**Status:** ✅ Implemented and tested

---

## What Was Implemented

### 1. Python Logging Module (`spimts/helpers/logging_config.py`)

**Architecture: Hybrid Dual-Output System**

- **StreamHandler (Terminal)**: Always active, color-coded, no timestamps
- **FileHandler (Optional)**: Only when `--log-file` provided, includes timestamps

**Why this design:**
- ✅ Real-time terminal visibility (no PowerShell buffering)
- ✅ User controls when/where logs are saved
- ✅ No special character encoding issues
- ✅ Structured format with timestamps in files
- ❌ NOT automatic (user must opt-in with `--log-file`)

**This is Option C+ (improved version of your preference)**

---

## Usage Examples

### Quick Test (No Log File)

```powershell
python -m spimts visualize --profile dev++ --models "Kuramoto"
```

**Result:**
- Immediate colored output in terminal
- No files created
- Progress visible in real-time

### Full Run with Auto-Named Log

```powershell
python -m spimts visualize --profile dev++ --all-runs --log-file auto
```

**Result:**
- Colored terminal output (real-time)
- Log file created: `.\logs\visualize_dev++_YYYYMMDD-HHMMSS.log`
- File includes timestamps

### Custom Log Location

```powershell
python -m spimts visualize --profile dev++ --all-runs --log-file "C:\MyLogs\run1.log"
```

---

## What Changed in visualize.py

### Before (Old Code)

```python
def main():
    args = p.parse_args()
    
    print(f"[VIZ] {model_safe} (run_id={rid})")
    # ... sparse output, no progress details ...
```

**Problems:**
- Only printed model names
- No progress indicators
- No file logging
- No timestamps

### After (New Code)

```python
from spimts.helpers.logging_config import setup_logging

def main():
    p.add_argument("--log-file", default=None, ...)
    args = p.parse_args()
    
    # Setup logging (new)
    logger = setup_logging('visualize', args.profile, args.log_file)
    
    logger.info(f"Found {len(runs)} model run(s) to visualize")
    logger.info(f"Profile: {args.profile} | MPI heatmaps: {args.include_mpi_heatmaps}")
    
    for idx, (rid, model_safe, model_dir, _mtime) in enumerate(runs, 1):
        logger.info(f"[{idx}/{len(runs)}] {model_safe} (run_id={rid})")
        logger.info(f"  Loaded {len(all_spi_names)} SPIs from cached artifacts")
        logger.info(f"  Saved MTS heatmap (T={X.shape[0]}, M={X.shape[1]})")
        logger.info(f"  Saved {len(mpi_spis_to_plot)} MPI heatmaps")
        logger.info(f"  Generating SPI-space plots (grid + individual)...")
        logger.info(f"  Saved 3 grid plots + {n} individual plots per method")
        logger.info(f"  Generating fingerprints and dendrograms...")
        logger.info(f"  Saved 3 fingerprints + 3 dendrograms")
        logger.info(f"  Completed visualization for {model_safe}")
    
    logger.info("All visualizations complete!")
```

**Improvements:**
- ✅ Detailed progress at each step
- ✅ Counts (SPIs, plots, heatmaps)
- ✅ Real-time updates
- ✅ Optional file logging
- ✅ Color-coded levels (INFO=green, WARNING=yellow, ERROR=red)

---

## Sample Output

### Terminal (Colored, Real-Time)

```
INFO     Logging to file: C:\...\logs\visualize_dev++_20251025-193601.log
INFO     Found 2 model run(s) to visualize
INFO     Profile: dev++ | MPI heatmaps: none
INFO     
INFO     [1/2] 25e48d9a_Kuramoto (run_id=20251025-081150)
INFO       Loaded 20 SPIs from cached artifacts
INFO       Saved MTS heatmap (T=7500, M=15)
INFO       Skipped MPI heatmaps (mode: none)
INFO       Generating SPI-space plots (grid + individual)...
INFO       Saved 3 grid plots + 190 individual plots per method
INFO       Generating fingerprints and dendrograms...
INFO       Saved 3 fingerprints + 3 dendrograms
INFO       Completed visualization for 25e48d9a_Kuramoto
INFO     
INFO     [2/2] e0eae877_Kuramoto (run_id=20251025-103003)
INFO       Loaded 20 SPIs from cached artifacts
...
INFO     All visualizations complete!
```

### Log File (Timestamps, Plain Text)

```
[19:36:01] INFO     Logging to file: C:\...\logs\visualize_dev++_20251025-193601.log
[19:36:01] INFO     Found 2 model run(s) to visualize
[19:36:01] INFO     Profile: dev++ | MPI heatmaps: none
[19:36:01] INFO     
[19:36:01] INFO     [1/2] 25e48d9a_Kuramoto (run_id=20251025-081150)
[19:36:01] INFO       Loaded 20 SPIs from cached artifacts
[19:36:02] INFO       Saved MTS heatmap (T=7500, M=15)
[19:36:02] INFO       Skipped MPI heatmaps (mode: none)
[19:36:02] INFO       Generating SPI-space plots (grid + individual)...
[19:38:44] INFO       Saved 3 grid plots + 190 individual plots per method
[19:38:44] INFO       Generating fingerprints and dendrograms...
[19:38:47] INFO       Saved 3 fingerprints + 3 dendrograms
[19:38:47] INFO       Completed visualization for 25e48d9a_Kuramoto
...
[19:41:23] INFO     All visualizations complete!
```

**Notice:**
- Terminal: No timestamps (cleaner, less clutter)
- File: Timestamps show exactly when each step occurred
- Both: Identical message content

---

## Test Results

**Command:**
```powershell
python -m spimts visualize --profile dev++ --models "Kuramoto" --include-mpi-heatmaps none --log-file auto
```

**Results:**
- ✅ Terminal output appeared in real-time (no buffering)
- ✅ Log file created: `logs\visualize_dev++_20251025-193601.log`
- ✅ Progress details show exactly what's happening
- ✅ Colors work in PowerShell
- ✅ Timestamps in file, clean format in terminal
- ✅ RuntimeWarnings (NumPy) still appear on stderr (expected, not from logger)

**Time tracking from log:**
- 19:36:01 - Started
- 19:38:44 - Finished first model (2 min 43 sec for 190 individual plots)
- 19:41:23 - Finished second model (5 min 22 sec total)

---

## Next Steps for compute.py

**Status:** Not yet implemented (visualize.py done first as proof-of-concept)

**When ready, same approach:**
1. Import `setup_logging`
2. Add `--log-file` argument
3. Replace `print()` statements with `logger.info()`, `logger.warning()`, etc.
4. Add progress details (current progress bar is good, keep it)

**Minimal changes needed:**
```python
# At top of compute.py
from spimts.helpers.logging_config import setup_logging

# In main()
ap.add_argument("--log-file", default=None, ...)
logger = setup_logging('compute', args.mode, args.log_file)

# Replace prints
logger.info(f"COMPUTE MODE: {args.mode} | Models: {total_models}")
logger.info(f"[{idx}/{total_models}] {model} M={M} T={T}")
logger.info(f"  [OK] {len(spi_names)} SPIs | {elapsed:.1f}s | ETA: {eta:.0f}s")
```

---

## Why This Design (Justification)

### Why NOT automatic logging to `./logs/`?

**Decision:** Only log to file when user provides `--log-file`

**Reasoning:**
1. **User control**: Quick tests don't need permanent logs
2. **Flexibility**: User might want logs elsewhere (network drive, cloud)
3. **No surprises**: User knows exactly when/where files are created
4. **Disk space**: Large runs can generate MB of logs

**Alternative rejected:** Auto-log to `./logs/` always
- **Problem:** User can't easily disable it
- **Problem:** Clutters disk with logs from every test run
- **Problem:** User might not want logs in that location

### Why dual output (terminal + file)?

**Decision:** Terminal output always visible, file is optional backup

**Reasoning:**
1. **Primary interface is terminal**: User needs real-time feedback
2. **File is archival**: For debugging later, not for watching live
3. **Separation**: Different formats optimized for different use cases

**Alternative rejected:** File-only logging with `--verbose` flag for terminal
- **Problem:** User must remember to add `--verbose` to see anything
- **Problem:** Defaults to silent mode (confusing UX)

### Why timestamps only in file?

**Decision:** Terminal shows `INFO message`, file shows `[HH:MM:SS] INFO message`

**Reasoning:**
1. **Terminal is ephemeral**: User watching live doesn't need exact times
2. **File is persistent**: Timestamps critical for debugging ("When did this fail?")
3. **Cleaner terminal**: Less visual clutter
4. **File parsability**: Easy to grep for time ranges

**Alternative rejected:** Timestamps everywhere
- **Problem:** Terminal output becomes cluttered
- **Problem:** Timestamps take visual space from actual message content

---

## Documentation Created

1. **`spimts/helpers/logging_config.py`** (108 lines)
   - `setup_logging()` function
   - `ColoredFormatter` class
   - `log_progress_bar()` helper (for future use)

2. **`LOGGING.md`** (370 lines)
   - Complete usage guide
   - API reference
   - Troubleshooting
   - Design rationale
   - Migration guide

3. **`LOGGING_IMPLEMENTATION_SUMMARY.md`** (this file)
   - Quick reference
   - Test results
   - Next steps

---

## Summary

**What you now have:**
- ✅ Real-time progress visibility in terminal (colored, immediate flush)
- ✅ Optional file logging (`--log-file auto` or explicit path)
- ✅ Detailed progress messages (SPIs loaded, plots saved, etc.)
- ✅ No PowerShell encoding issues
- ✅ Structured format (timestamps in files, clean in terminal)
- ✅ Tested and working on visualize.py

**What's NOT done:**
- ⏳ compute.py logging (same approach, trivial to add when needed)
- ⏳ Progress bars (helper function exists, not yet used)

**Recommended workflow:**
```powershell
# Quick test (no log)
python -m spimts visualize --profile dev++ --models "Kuramoto"

# Full run (with log)
python -m spimts visualize --profile dev++ --all-runs --log-file auto

# Check log afterward
Get-Content .\logs\visualize_dev++_*.log | Select-String "ERROR|WARNING"
```

**Your original concerns addressed:**
- ✅ "Real problem is lack of real-time progress visibility" → **SOLVED** (immediate terminal output)
- ✅ "Don't see anything until the end or when errors occur" → **SOLVED** (progress at each step)
- ✅ "RuntimeWarnings clutter output" → **EXPLAINED** (not from logger, expected behavior)
- ✅ "How to check what completed so far" → **SOLVED** (log file has timestamps, terminal shows progress)
