#!/usr/bin/env python3
"""
Script to check for errors in event processing and mark them in event_status.csv.

This script reads the event_status.csv file
2. Checks which events have figures in figures_0 and figures_A directories
3. For events that don't have figures, checks the error files for actual errors
4. Updates the CSV with error information

Usage:
    python check_event_errors.py              # Check all runs and update the CSV file
    python check_event_errors.py -r 0         # Check only Run 0
    python check_event_errors.py -r A         # Check only Run A

The script maps:
- figures_0/ directory ↔ outdir_0/ (Run 0)
- figures_A/ directory ↔ outdir_A/ (Run A)

For each event:
- If figure exists: preserves existing content (doesn't overwrite any existing scores)
- If figure missing but error file exists: extracts and marks the error
- If both figure and error file missing: marks as "Missing (no error file)"

## Error Message Types:
- **[Preserved]**: Event processed successfully, figure generated (existing content preserved)
- **Missing bilby data**: Required bilby data files not found
- **PSD data contains NaNs**: Power spectral density data has invalid values
- **GWOSC data unavailable**: Gravitational wave data not available from GWOSC
- **JAX internal error**: JAX framework internal computation error
- **Timeout error**: Process timed out during execution
- **Missing (no error file)**: No figure and no error file found

## File Management:
- Uses the existing `paths.py` module for consistent path handling
- Handles both directory mappings:
  - `figures_0/` ↔ `outdir_0/` (Run 0)
  - `figures_A/` ↔ `outdir_A/` (Run A)
- Can process specific runs or all runs depending on command-line options
"""

import os
import csv
import re
from pathlib import Path
import paths


def extract_error_message(error_file_path):
    """
    Extract the main error message from an error file.
    Looks for common error patterns like Traceback, Exception, etc.
    Returns a short, descriptive error message.
    """
    try:
        with open(error_file_path, 'r') as f:
            content = f.read()
        
        # Look for traceback and get the last error
        if 'Traceback' in content:
            lines = content.split('\n')
            traceback_started = False
            error_lines = []
            
            for line in lines:
                if 'Traceback' in line:
                    traceback_started = True
                    error_lines = []  # Reset to capture the last traceback
                elif traceback_started:
                    error_lines.append(line)
            
            # Find the actual error message (usually the last non-empty line)
            for line in reversed(error_lines):
                line = line.strip()
                if line and not line.startswith('  '):
                    # Clean up the error message and make it more concise
                    if ':' in line:
                        error_type, error_msg = line.split(':', 1)
                        error_type = error_type.strip()
                        error_msg = error_msg.strip()
                        
                        # Shorten common error types and messages
                        if error_type == "FileNotFoundError":
                            if "bilby_runs/outdir" in error_msg:
                                return "Missing bilby data"
                            return "File not found"
                        elif error_type == "ValueError":
                            if "PSD data" in error_msg and "NaNs" in error_msg:
                                return "PSD data contains NaNs"
                            elif "Cannot find a GWOSC dataset" in error_msg:
                                return "GWOSC data unavailable"
                            return f"ValueError: {error_msg[:50]}..." if len(error_msg) > 50 else f"ValueError: {error_msg}"
                        elif error_type == "TimeoutError":
                            return "Timeout error"
                        elif "JAX has removed its internal frames" in line:
                            return "JAX internal error"
                        else:
                            return f"{error_type}: {error_msg[:50]}..." if len(error_msg) > 50 else f"{error_type}: {error_msg}"
                    elif "JAX has removed its internal frames" in line:
                        return "JAX internal error"
                    return line[:80] + "..." if len(line) > 80 else line
        
        # Look for other error patterns
        error_patterns = [
            r'ERROR:.*',
            r'Error:.*',
            r'FAILED:.*', 
            r'Failed:.*',
            r'.*Exception:.*',
            r'.*Error:.*'
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Return the last (most recent) error, truncated if too long
                error = matches[-1].strip()
                return error[:80] + "..." if len(error) > 80 else error
        
        return "Unknown error"
    
    except Exception as e:
        return f"Could not read error file: {e}"


def check_event_status(specific_run=None):
    """
    Main function to check event status and update CSV with errors.
    
    Args:
        specific_run (str): If specified, only check this run (e.g., '0', 'A', 'B')
    """
    # Define paths
    csv_file = paths.static / "event_status.csv"
    
    # Determine which runs to check
    runs_to_check = []
    if specific_run:
        # Check if the specific run exists in the CSV header
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            run_column = f"Run {specific_run}"
            if run_column not in fieldnames:
                print(f"Error: Column '{run_column}' not found in CSV. Available columns: {fieldnames}")
                return
        runs_to_check = [specific_run]
    else:
        # Auto-detect available runs from CSV columns
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for field in fieldnames:
                if field.startswith("Run "):
                    run_suffix = field.replace("Run ", "")
                    runs_to_check.append(run_suffix)
    
    print(f"Checking runs: {runs_to_check}")
    
    # Get figure directories and outdirs for each run
    run_data = {}
    for run in runs_to_check:
        figures_dir = paths.src / f"figures_{run}"
        outdir = paths.static / "jim_runs" / f"outdir_{run}"
        
        figures_events = set()
        if figures_dir.exists():
            figures_events = {f.stem for f in figures_dir.glob("*.jpg")}
        
        run_data[run] = {
            'figures_dir': figures_dir,
            'outdir': outdir,
            'figures_events': figures_events
        }
        
        print(f"Found {len(figures_events)} figures in figures_{run}")
    
    # Read the CSV file
    updated_rows = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            event = row['Event']
            
            for run in runs_to_check:
                run_column = f"Run {run}"
                data = run_data[run]
                
                if event not in data['figures_events']:
                    # Check if there's an error file
                    error_file = data['outdir'] / event / f"{event}.err"
                    if error_file.exists():
                        error_msg = extract_error_message(error_file)
                        row[run_column] = f"ERROR: {error_msg}"
                        print(f"Run {run} - {event}: {error_msg}")
                    else:
                        row[run_column] = "Missing (no error file)"
                        print(f"Run {run} - {event}: Missing (no error file)")
                else:
                    # Don't change existing content for successful events (preserve any existing scores)
                    pass
            
            updated_rows.append(row)
    
    # Write the updated CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    
    print(f"Updated {csv_file}")
    
    # Print summary for each run
    print("\n=== SUMMARY ===")
    for run in runs_to_check:
        run_column = f"Run {run}"
        errors = sum(1 for row in updated_rows if row[run_column].startswith('ERROR'))
        missing = sum(1 for row in updated_rows if row[run_column].startswith('Missing'))
        preserved = sum(1 for row in updated_rows if not row[run_column].startswith('ERROR') and not row[run_column].startswith('Missing'))
        print(f"Run {run}: {preserved} preserved (successful), {errors} errors, {missing} missing")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check for errors in event processing and mark them in event_status.csv"
    )
    parser.add_argument(
        "-r", "--run", 
        type=str,
        help="Specify which run to check (e.g., '0', 'A', 'B'). If not specified, checks all runs."
    )
    
    args = parser.parse_args()
    
    if args.run:
        print(f"Checking only Run {args.run}")
    
    check_event_status(specific_run=args.run)
