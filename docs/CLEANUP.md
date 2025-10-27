# Cleanup & Inspection Guide

**Last Updated:** October 25, 2025

---

## Philosophy: PowerShell for Deletion, Python for Inspection

**Why not a Python cleanup script?**
- PowerShell already does deletion perfectly with `-WhatIf` dry-run, wildcards, filters
- Python adds unnecessary complexity (confirmation prompts, path validation, etc.)
- **Use the right tool for the job**: PowerShell for file operations, Python for metadata parsing

**What Python IS good for:**
- Inspecting what's been computed (reads `meta.json`)
- Finding duplicates (parses timestamps from folder names)
- Checking visualization status (counts plot types)

---

## Part 1: Inspection (Python)

### Show All Runs in a Profile

```powershell
python -m spimts.helpers.inspect list-runs --profile dev++
```

**Output:**
```
16 runs in 'dev++':

timestamp            model               M     T  n_spis  size_mb
2025-10-25 18:31:08  Kuramoto           15  7500      20    156.3
2025-10-25 18:24:31  TimeWarp-clones    15  2000      20     45.2
2025-10-25 18:22:30  GBM-returns        15  3000      20     54.8
...
```

### Show Recent Runs (Last N Days)

```powershell
# Last 7 days (default)
python -m spimts.helpers.inspect recent --profile dev++

# Last 2 days
python -m spimts.helpers.inspect recent --profile dev++ --days 2
```

### Summary Statistics

```powershell
python -m spimts.helpers.inspect summary --profile dev++
```

**Output:**
```
=== Summary for 'dev++' ===
Total runs: 16
Total size: 2.34 GB
Unique models: 10
Oldest run: 2025-10-25 17:00:17
Newest run: 2025-10-25 18:31:08
Avg SPIs/run: 20.0

Runs per model:
  CML-logistic              1 run(s)
  GBM-returns               1 run(s)
  Kuramoto                  2 run(s)
  Lorenz-96                 2 run(s)
  ...
```

### Find Duplicate Runs (Same Model, Different Timestamps)

```powershell
python -m spimts.helpers.inspect duplicates --profile dev++
```

**Output:**
```
=== Duplicate runs in 'dev++' ===

Kuramoto (2 runs):
  [2025-10-25 18:31] 156.3 MB  (age: 0 days)  <- NEWEST
  [2025-10-25 10:30]  45.1 MB  (age: 0 days)

VAR(1) (2 runs):
  [2025-10-25 17:00]  42.7 MB  (age: 0 days)  <- NEWEST
  [2025-10-24 16:43]  41.9 MB  (age: 1 days)
```

**Use case:** Identify old runs you can safely delete

### Check Visualization Status

```powershell
python -m spimts.helpers.inspect viz-status --profile dev++
```

**Output:**
```
=== Visualization status for 'dev++' ===

model               has_plots  plot_count
Kuramoto                 True        1150
TimeWarp-clones          True        1150
GBM-returns              True        1150
VAR(1)                  False           0

Total: 15/16 runs have plots
```

**Use case:** Find runs that failed visualization or are incomplete

---

## Part 2: Cleanup (PowerShell)

### Safety First: Always Use `-WhatIf` Dry-Run

```powershell
# Preview what will be deleted (SAFE - nothing actually deleted)
Remove-Item results\dev\* -Recurse -WhatIf
```

### Delete Entire Profile

```powershell
# Delete all 'dev' runs
Remove-Item results\dev\* -Recurse -Force

# Delete all 'dev+' runs
Remove-Item results\dev+\* -Recurse -Force
```

### Delete Specific Model Runs

```powershell
# Delete all Kuramoto runs in dev++
Remove-Item results\dev++\*Kuramoto -Recurse -Force

# Delete all VAR(1) runs in dev++
Remove-Item results\dev++\*VAR* -Recurse -Force
```

### Delete Old Runs (By Timestamp)

```powershell
# Delete runs older than 7 days in dev++
Get-ChildItem results\dev++\* -Directory | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | 
    Remove-Item -Recurse -Force

# Preview first (dry-run)
Get-ChildItem results\dev++\* -Directory | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | 
    Select-Object Name, LastWriteTime
```

### Delete Only Plots (Keep Computed Data)

```powershell
# Delete all plots in dev++ (keep arrays/csv)
Remove-Item results\dev++\*\plots -Recurse -Force

# Delete only SPI-space individual plots (large files)
Remove-Item results\dev++\*\plots\spi_space_individual -Recurse -Force

# Delete only MPI heatmaps
Remove-Item results\dev++\*\plots\mpis -Recurse -Force
```

### Delete Duplicate Runs (Keep Only Newest)

**Step 1:** Find duplicates with Python:
```powershell
python -m spimts.helpers.inspect duplicates --profile dev++
```

**Step 2:** Manually delete older runs:
```powershell
# Example: Kuramoto has 2 runs, delete the older one
# (Inspect output showed: 2025-10-25 10:30 is older)
Remove-Item results\dev++\20251025-103003_e0eae877_Kuramoto -Recurse -Force
```

**OR** use PowerShell to auto-delete older duplicates:
```powershell
# Group by model, keep only newest per model
Get-ChildItem results\dev++\* -Directory | 
    Group-Object { ($_.Name -split '_', 3)[2] } | 
    ForEach-Object {
        # Sort by timestamp (newest first), skip first, delete rest
        $_.Group | Sort-Object Name -Descending | Select-Object -Skip 1 | Remove-Item -Recurse -Force
    }
```

**Always preview first:**
```powershell
Get-ChildItem results\dev++\* -Directory | 
    Group-Object { ($_.Name -split '_', 3)[2] } | 
    ForEach-Object {
        $newest = $_.Group | Sort-Object Name -Descending | Select-Object -First 1
        $toDelete = $_.Group | Sort-Object Name -Descending | Select-Object -Skip 1
        Write-Host "Model: $($_.Name)"
        Write-Host "  KEEP: $($newest.Name)"
        $toDelete | ForEach-Object { Write-Host "  DELETE: $($_.Name)" }
    }
```

### Delete Cache Files

```powershell
# Delete all cached calc_table files
Remove-Item cache\* -Force

# Delete only specific model cache
Remove-Item cache\*Kuramoto* -Force

# Delete only old cache files (>7 days)
Get-ChildItem cache\* | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item -Force
```

### Selective Deletion by Size

```powershell
# Find large runs (>100 MB)
Get-ChildItem results\dev++\* -Directory | 
    ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        [PSCustomObject]@{
            Name = $_.Name
            SizeMB = [math]::Round($size, 1)
        }
    } | Where-Object { $_.SizeMB -gt 100 } | Format-Table

# Delete runs larger than 200 MB
Get-ChildItem results\dev++\* -Directory | 
    Where-Object {
        $size = (Get-ChildItem $_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        $size -gt 200
    } | Remove-Item -Recurse -Force
```

---

## Part 3: Common Workflows

### Workflow 1: Clean Dev After Moving to Dev++

**Goal:** Free up space from `dev` profile after successful `dev++` runs

```powershell
# 1. Verify dev++ has all models
python -m spimts.helpers.inspect summary --profile dev++

# 2. Check dev size
python -m spimts.helpers.inspect summary --profile dev

# 3. Delete all dev runs
Remove-Item results\dev\* -Recurse -Force
```

### Workflow 2: Delete Old Duplicates

**Goal:** Keep only newest run per model in `dev++`

```powershell
# 1. Find duplicates
python -m spimts.helpers.inspect duplicates --profile dev++

# 2. Preview deletion (dry-run)
Get-ChildItem results\dev++\* -Directory | 
    Group-Object { ($_.Name -split '_', 3)[2] } | 
    ForEach-Object {
        $newest = $_.Group | Sort-Object Name -Descending | Select-Object -First 1
        $toDelete = $_.Group | Sort-Object Name -Descending | Select-Object -Skip 1
        Write-Host "Model: $($_.Name)"
        Write-Host "  KEEP: $($newest.Name)"
        $toDelete | ForEach-Object { Write-Host "  DELETE: $($_.Name)" }
    }

# 3. Execute deletion
Get-ChildItem results\dev++\* -Directory | 
    Group-Object { ($_.Name -split '_', 3)[2] } | 
    ForEach-Object {
        $_.Group | Sort-Object Name -Descending | Select-Object -Skip 1 | Remove-Item -Recurse -Force
    }

# 4. Verify
python -m spimts.helpers.inspect duplicates --profile dev++
# Should show: "No duplicate runs found"
```

### Workflow 3: Free Up Space by Deleting Plots Only

**Goal:** Keep computed SPIs but delete large plot files (can regenerate later)

```powershell
# 1. Check current size
python -m spimts.helpers.inspect summary --profile dev++

# 2. Delete individual plots (largest files)
Remove-Item results\dev++\*\plots\spi_space_individual -Recurse -Force

# 3. Check new size
python -m spimts.helpers.inspect summary --profile dev++

# 4. Regenerate plots later if needed
python -m spimts visualize --profile dev++ --all-runs --include-mpi-heatmaps none
```

### Workflow 4: Fresh Start (Nuclear Option)

**Goal:** Delete everything and start clean

```powershell
# 1. Backup important runs (optional)
Copy-Item results\dev++\*Kuramoto* -Destination C:\Backup\ -Recurse

# 2. Delete all results and cache
Remove-Item results\* -Recurse -Force
Remove-Item cache\* -Force

# 3. Recreate directory structure
New-Item -ItemType Directory -Path results\dev, results\dev+, results\dev++, results\paper, cache -Force

# 4. Verify
python -m spimts.helpers.inspect summary --profile dev++
# Should show: "Total runs: 0"
```

---

## Part 4: Advanced Recipes

### Find Runs Without Visualizations

```powershell
# 1. Check viz status
python -m spimts.helpers.inspect viz-status --profile dev++ --format json > viz_status.json

# 2. Use PowerShell to filter
Get-Content viz_status.json | ConvertFrom-Json | Where-Object { -not $_.has_plots } | 
    ForEach-Object { $_.run_id }

# 3. Delete runs without plots (if you want)
Get-Content viz_status.json | ConvertFrom-Json | Where-Object { -not $_.has_plots } | 
    ForEach-Object {
        $pattern = $_.run_id.Replace('-', '?')  # Wildcards for partial match
        Remove-Item results\dev++\*$pattern* -Recurse -Force
    }
```

### Export Inventory to CSV

```powershell
python -m spimts.helpers.inspect list-runs --profile dev++ --format json | 
    ConvertFrom-Json | 
    Export-Csv -Path inventory_dev++.csv -NoTypeInformation
```

### Calculate Disk Usage by Model

```powershell
Get-ChildItem results\dev++\* -Directory | 
    Group-Object { ($_.Name -split '_', 3)[2] } | 
    ForEach-Object {
        $totalSize = ($_.Group | Get-ChildItem -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        [PSCustomObject]@{
            Model = $_.Name
            Runs = $_.Count
            TotalMB = [math]::Round($totalSize, 1)
            AvgMB = [math]::Round($totalSize / $_.Count, 1)
        }
    } | Sort-Object TotalMB -Descending | Format-Table
```

---

## Safety Guidelines

**Always:**
1. ✅ Use `-WhatIf` before `Remove-Item` to preview
2. ✅ Inspect with Python first (`inspect duplicates`, `inspect viz-status`)
3. ✅ Start with small deletions (single model) before bulk operations
4. ✅ Check `summary` before and after cleanup to verify

**Never:**
1. ❌ Delete results during active computation (check for running Python processes)
2. ❌ Delete cache files while pyspi Calculator is running
3. ❌ Use `-Force` without testing `-WhatIf` first

**Recovery:**
- Deleted results: No recovery (must re-compute)
- Deleted plots: Easy recovery (run `visualize` again on cached arrays)
- Deleted cache: Minor recovery (pyspi can recalculate, but slower)

---

## Quick Reference

| Task | Command |
|------|---------|
| List all runs | `python -m spimts.helpers.inspect list-runs --profile dev++` |
| Show summary | `python -m spimts.helpers.inspect summary --profile dev++` |
| Find duplicates | `python -m spimts.helpers.inspect duplicates --profile dev++` |
| Check viz status | `python -m spimts.helpers.inspect viz-status --profile dev++` |
| Delete entire profile | `Remove-Item results\dev\* -Recurse -Force` |
| Delete specific model | `Remove-Item results\dev++\*Kuramoto* -Recurse -Force` |
| Delete only plots | `Remove-Item results\dev++\*\plots -Recurse -Force` |
| Delete old runs (>7 days) | `Get-ChildItem results\dev++\* -Directory \| Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } \| Remove-Item -Recurse -Force` |
| Preview deletion | Add `-WhatIf` to any `Remove-Item` command |

---

## Why This Design?

**Python for inspection:**
- Parses `meta.json` (M, T, n_spis)
- Extracts timestamps from folder names
- Calculates statistics (avg SPIs, total size)
- Groups duplicates intelligently

**PowerShell for deletion:**
- Built-in `-WhatIf` dry-run
- Native file operations (fast)
- Powerful filtering (`Where-Object`, wildcards)
- No confirmation prompts needed (user controls with `-WhatIf`)

**Result:** Best of both worlds - Python for metadata, PowerShell for operations.
