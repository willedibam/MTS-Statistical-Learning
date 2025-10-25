# Windows Power Settings for Overnight Run

## CRITICAL: Configure Before Running

Python scripts **CANNOT** prevent Windows from sleeping. You must manually configure power settings to ensure the overnight run completes without interruption.

---

## Required Configuration

### Option 1: PowerShell (Run as Administrator)

Open PowerShell **as Administrator** and run:

```powershell
# Disable sleep on AC power (laptop plugged in / desktop always)
powercfg /change standby-timeout-ac 0

# Optional: Turn off display after 30 minutes (saves energy without interrupting compute)
powercfg /change monitor-timeout-ac 30

# Verify settings
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
```

**Expected output**:
```
Current AC Power Setting Index: 0x00000000  # 0 = Never sleep
```

---

### Option 2: Windows Settings GUI

1. Open **Settings** → **System** → **Power & sleep**
2. Under **Screen**, set:
   - **On battery power, turn off after**: 30 minutes (or your preference)
   - **When plugged in, turn off after**: 30 minutes
3. Under **Sleep**, set:
   - **On battery power, PC goes to sleep after**: Never
   - **When plugged in, PC goes to sleep after**: **Never** ← **CRITICAL**
4. Click **Additional power settings** (opens Control Panel)
5. Select your current power plan → **Change plan settings**
6. Set **Put the computer to sleep**: **Never**
7. Click **Change advanced power settings**
8. Expand **Sleep** → **Sleep after**:
   - **Plugged in**: **0** (Never) ← **CRITICAL**
9. Click **OK** → **Save changes**

---

## Verification

Run this PowerShell command to verify:

```powershell
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String -Pattern "Current AC Power Setting Index:"
```

**Should output**:
```
Current AC Power Setting Index: 0x00000000
```

If you see anything other than `0x00000000`, **sleep is still enabled** and the run may be interrupted.

---

## Restore Settings After Run

After the overnight run completes, restore normal power settings:

```powershell
# Re-enable sleep after 30 minutes on AC
powercfg /change standby-timeout-ac 30

# Verify
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
```

Or use Windows Settings GUI to restore your preferred sleep timeout.

---

## Why This Matters

- **Expected runtime**: 8-12 hours for dev+++ profile (15 models × 30-60 min each)
- **Computer sleep**: Suspends all processes, including Python
- **Result**: Overnight run stops mid-computation, produces incomplete results
- **Data loss risk**: Partial runs may corrupt cache or results directories

**Bottom line**: If the computer sleeps, the run WILL fail. Configure power settings before starting.

---

## Additional Recommendations

### 1. Laptop Users
- **Plug in AC adapter** before starting (don't run on battery)
- **Close laptop lid settings**: Set "When I close the lid" → "Do nothing" (Settings → System → Power & sleep → Additional power settings → Choose what closing the lid does)

### 2. Desktop Users
- Ensure **no scheduled maintenance** or Windows Update restarts during run (Settings → Update & Security → Windows Update → Advanced options → "Pause updates")

### 3. All Users
- **Disable screensaver password**: Screensaver won't interrupt compute, but password prompt can cause issues with monitoring
- **Close unnecessary applications**: Free up RAM (dev+++ needs ~4-8 GB depending on SPIs)
- **Monitor first 30-60 minutes**: Ensure first model completes successfully before leaving computer unattended

---

## Troubleshooting

### Computer still sleeps despite settings
- **Check active power plan**: `powercfg /getactivescheme`
- **Modify correct plan**: Settings only apply to the **active** plan
- **Hybrid sleep**: Disable in Advanced power settings → Sleep → Allow hybrid sleep → Off

### Run stops after few hours
- **Check Windows Event Viewer**: Search for "Sleep" events (Event Viewer → Windows Logs → System)
- **Check for automatic restarts**: Windows Update may restart without warning if updates pending

### Can't run PowerShell as Administrator
- Right-click **Start menu** → **Windows PowerShell (Admin)**
- If unavailable, search "PowerShell" → Right-click → "Run as administrator"

---

## Summary Checklist

Before running `overnight_run.ps1`:

- [ ] AC power connected (laptops)
- [ ] Sleep timeout set to **Never** on AC power (`powercfg /change standby-timeout-ac 0`)
- [ ] Verified with `powercfg /query` → `0x00000000`
- [ ] Windows Update paused (no automatic restarts)
- [ ] Close lid action set to "Do nothing" (laptops)
- [ ] ~10 GB free disk space available
- [ ] Python and pyspi installed and working

Once configured, run:

```powershell
.\overnight_run.ps1
```

Monitor first model completion (~30-60 min), then leave overnight.

---

**Document version**: 1.0  
**Last updated**: January 2025  
**Applies to**: Windows 10/11, PowerShell 5.1+
