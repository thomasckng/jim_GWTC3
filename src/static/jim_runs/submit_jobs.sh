#!/bin/bash

# Define usage
usage() {
    echo "Usage: $0 [-o outdir] (use -o to specify output directory)"
    exit 1
}

# Default to no outdir specified
OUTDIR=""

# Parse command line options
while getopts "o:" opt; do
    case $opt in
        o) OUTDIR=$OPTARG ;;
        ?) usage ;;
    esac
done

# Check if outdir is specified. If not, use "outdir" as the default.
if [ -z "$OUTDIR" ]; then
    OUTDIR="outdir"
fi

# Define the path to the template script
template_file="template.sh"

# Create directories to store the SLURM scripts and logs
mkdir -p slurm_scripts
mkdir -p logs
mkdir -p $OUTDIR

# Read the event IDs from the CSV file
csv_file="../event_status.csv"
gw_ids=$(awk -F, 'NR>1 {print $1}' $csv_file)

# Loop over each GW event ID
for gw_id in $gw_ids
do
  # Define the result directory path
  result_dir="$OUTDIR"
  
  # Check if the result directory contains any files
  if [ -d "$result_dir" ] && [ "$(find "$result_dir" -type f | wc -l)" -gt 0 ]; then
    continue
  fi
  
  # Create a unique SLURM script for each GW event
  new_script="slurm_scripts/submit_${gw_id}.sh"
  cp $template_file $new_script
  
  # Replace the placeholder with the actual GW_ID
  sed -i "s/{{{GW_ID}}}/$gw_id/g" $new_script

  # Replace the placeholder with the actual outdir
  sed -i "s#default#$result_dir#g" $new_script

  # Make the script executable
  chmod +x $new_script

  # Submit
  sbatch $new_script
  echo "Submitted job for $gw_id"
done
