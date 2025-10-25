# spimts/helpers/inspect.py
"""
Inspection utilities for cached results and visualizations.

This module provides functions to inspect what has been computed/visualized,
without deletion capabilities (use PowerShell for that - it's better at it).

Usage:
    python -m spimts.helpers.inspect list-runs --profile dev++
    python -m spimts.helpers.inspect recent --profile dev++ --days 7
    python -m spimts.helpers.inspect summary --profile dev++
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


def _parse_run_dir(name: str) -> tuple[Optional[str], Optional[str], Optional[datetime]]:
    """Extract (run_id, model, timestamp) from folder name like '20251025-081150_25e48d9a_Kuramoto'"""
    parts = name.split('_', 2)
    if len(parts) < 3:
        return None, None, None
    
    timestamp_str, run_hash, model = parts[0], parts[1], parts[2]
    try:
        # Parse YYYYMMDD-HHMMSS format
        ts = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
        return f"{timestamp_str}_{run_hash}", model, ts
    except ValueError:
        return None, None, None


def list_runs(profile: str, root: str = "./results") -> List[Dict]:
    """List all runs in a profile with metadata.
    
    Returns:
        List of dicts with keys: run_id, model, timestamp, path, size_mb, n_spis
    """
    base = Path(root) / profile
    if not base.exists():
        return []
    
    runs = []
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        
        run_id, model, ts = _parse_run_dir(folder.name)
        if run_id is None:
            continue
        
        # Load meta.json if exists
        meta_path = folder / "meta.json"
        n_spis = None
        M, T = None, None
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                    n_spis = meta.get('n_spis')
                    M = meta.get('M')
                    T = meta.get('T')
            except:
                pass
        
        # Calculate folder size
        size_bytes = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
        size_mb = size_bytes / (1024 * 1024)
        
        runs.append({
            'run_id': run_id,
            'model': model,
            'timestamp': ts,
            'path': str(folder),
            'size_mb': size_mb,
            'n_spis': n_spis,
            'M': M,
            'T': T
        })
    
    return sorted(runs, key=lambda r: r['timestamp'], reverse=True)


def recent_runs(profile: str, days: int = 7, root: str = "./results") -> List[Dict]:
    """Get runs from last N days."""
    all_runs = list_runs(profile, root)
    cutoff = datetime.now() - timedelta(days=days)
    return [r for r in all_runs if r['timestamp'] >= cutoff]


def summarize_profile(profile: str, root: str = "./results") -> Dict:
    """Get summary statistics for a profile."""
    runs = list_runs(profile, root)
    
    if not runs:
        return {'profile': profile, 'total_runs': 0}
    
    df = pd.DataFrame(runs)
    
    summary = {
        'profile': profile,
        'total_runs': len(runs),
        'total_size_gb': df['size_mb'].sum() / 1024,
        'models': df['model'].nunique(),
        'model_counts': df['model'].value_counts().to_dict(),
        'oldest_run': df['timestamp'].min(),
        'newest_run': df['timestamp'].max(),
        'avg_spis_per_run': df['n_spis'].mean() if df['n_spis'].notna().any() else None,
    }
    
    return summary


def find_duplicates(profile: str, root: str = "./results") -> Dict[str, List[Dict]]:
    """Find duplicate runs (same model, different timestamps).
    
    Returns:
        Dict mapping model name to list of runs (sorted newest first)
    """
    runs = list_runs(profile, root)
    
    # Group by model
    by_model = {}
    for run in runs:
        model = run['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(run)
    
    # Keep only models with multiple runs
    duplicates = {m: sorted(runs_list, key=lambda r: r['timestamp'], reverse=True) 
                  for m, runs_list in by_model.items() if len(runs_list) > 1}
    
    return duplicates


def check_visualizations(profile: str, root: str = "./results") -> List[Dict]:
    """Check which runs have visualizations completed.
    
    Returns:
        List of dicts with: run_id, model, has_plots, plot_types, plot_count
    """
    runs = list_runs(profile, root)
    
    results = []
    for run in runs:
        folder = Path(run['path'])
        plots_dir = folder / "plots"
        
        has_plots = plots_dir.exists()
        plot_types = []
        plot_count = 0
        
        if has_plots:
            # Check for different plot types
            if (plots_dir / "spi_space").exists():
                plot_types.append("spi_space")
                plot_count += len(list((plots_dir / "spi_space").glob("*.png")))
            
            if (plots_dir / "fingerprint").exists():
                plot_types.append("fingerprint")
                plot_count += len(list((plots_dir / "fingerprint").glob("*.png")))
            
            if (plots_dir / "mpis").exists():
                plot_types.append("mpis")
                plot_count += len(list((plots_dir / "mpis").glob("*.png")))
            
            if (plots_dir / "spi_space_individual").exists():
                plot_types.append("individual")
                # Count all individual plots across methods
                for method_dir in (plots_dir / "spi_space_individual").iterdir():
                    if method_dir.is_dir():
                        plot_count += len(list(method_dir.glob("*.png")))
        
        results.append({
            'run_id': run['run_id'],
            'model': run['model'],
            'has_plots': has_plots,
            'plot_types': plot_types,
            'plot_count': plot_count
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Inspect computed results and visualizations")
    parser.add_argument('command', choices=['list-runs', 'recent', 'summary', 'duplicates', 'viz-status'],
                       help="Inspection command to run")
    parser.add_argument('--profile', choices=['dev', 'dev+', 'dev++', 'paper'], default='dev++',
                       help="Profile to inspect")
    parser.add_argument('--root', default='./results', help="Results root directory")
    parser.add_argument('--days', type=int, default=7, help="Days for 'recent' command")
    parser.add_argument('--format', choices=['table', 'json'], default='table',
                       help="Output format")
    
    args = parser.parse_args()
    
    if args.command == 'list-runs':
        runs = list_runs(args.profile, args.root)
        if args.format == 'json':
            # Convert datetime to string for JSON
            for r in runs:
                r['timestamp'] = r['timestamp'].isoformat() if r['timestamp'] else None
            import json
            print(json.dumps(runs, indent=2))
        else:
            df = pd.DataFrame(runs)
            if not df.empty:
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                df['size_mb'] = df['size_mb'].round(1)
                print(f"\n{len(runs)} runs in '{args.profile}':\n")
                print(df[['timestamp', 'model', 'M', 'T', 'n_spis', 'size_mb']].to_string(index=False))
            else:
                print(f"No runs found in '{args.profile}'")
    
    elif args.command == 'recent':
        runs = recent_runs(args.profile, args.days, args.root)
        if args.format == 'json':
            for r in runs:
                r['timestamp'] = r['timestamp'].isoformat() if r['timestamp'] else None
            import json
            print(json.dumps(runs, indent=2))
        else:
            df = pd.DataFrame(runs)
            if not df.empty:
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                df['size_mb'] = df['size_mb'].round(1)
                print(f"\n{len(runs)} runs from last {args.days} days in '{args.profile}':\n")
                print(df[['timestamp', 'model', 'M', 'T', 'n_spis', 'size_mb']].to_string(index=False))
            else:
                print(f"No runs from last {args.days} days in '{args.profile}'")
    
    elif args.command == 'summary':
        summary = summarize_profile(args.profile, args.root)
        if args.format == 'json':
            # Convert datetime to string
            summary['oldest_run'] = summary.get('oldest_run').isoformat() if summary.get('oldest_run') else None
            summary['newest_run'] = summary.get('newest_run').isoformat() if summary.get('newest_run') else None
            import json
            print(json.dumps(summary, indent=2))
        else:
            print(f"\n=== Summary for '{args.profile}' ===")
            print(f"Total runs: {summary['total_runs']}")
            print(f"Total size: {summary['total_size_gb']:.2f} GB")
            print(f"Unique models: {summary['models']}")
            if summary.get('oldest_run'):
                print(f"Oldest run: {summary['oldest_run'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Newest run: {summary['newest_run'].strftime('%Y-%m-%d %H:%M:%S')}")
            if summary.get('avg_spis_per_run'):
                print(f"Avg SPIs/run: {summary['avg_spis_per_run']:.1f}")
            print(f"\nRuns per model:")
            for model, count in sorted(summary['model_counts'].items()):
                print(f"  {model:25s} {count:3d} run(s)")
    
    elif args.command == 'duplicates':
        dupes = find_duplicates(args.profile, args.root)
        if args.format == 'json':
            # Convert datetime to string
            for runs_list in dupes.values():
                for r in runs_list:
                    r['timestamp'] = r['timestamp'].isoformat() if r['timestamp'] else None
            import json
            print(json.dumps(dupes, indent=2))
        else:
            if not dupes:
                print(f"No duplicate runs found in '{args.profile}'")
            else:
                print(f"\n=== Duplicate runs in '{args.profile}' ===\n")
                for model, runs_list in sorted(dupes.items()):
                    print(f"{model} ({len(runs_list)} runs):")
                    for r in runs_list:
                        age_days = (datetime.now() - r['timestamp']).days
                        print(f"  [{r['timestamp'].strftime('%Y-%m-%d %H:%M')}] {r['size_mb']:6.1f} MB  (age: {age_days} days)")
                    print()
    
    elif args.command == 'viz-status':
        status = check_visualizations(args.profile, args.root)
        if args.format == 'json':
            import json
            print(json.dumps(status, indent=2))
        else:
            df = pd.DataFrame(status)
            if not df.empty:
                print(f"\n=== Visualization status for '{args.profile}' ===\n")
                print(df[['model', 'has_plots', 'plot_count']].to_string(index=False))
                print(f"\nTotal: {df['has_plots'].sum()}/{len(df)} runs have plots")
            else:
                print(f"No runs found in '{args.profile}'")


if __name__ == '__main__':
    main()
