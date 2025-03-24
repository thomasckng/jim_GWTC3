#!/bin/bash

# Define usage
usage() {
    echo "Usage: $0 [-n] [-o outdir] (use -n to enable node preference for a/c nodes, -o to specify output directory)"
    exit 1
}

# Default to no node preference and no outdir specified
USE_NODE_PREFERENCE=false
OUTDIR=""

# Parse command line options
while getopts "no:" opt; do
    case $opt in
        n) USE_NODE_PREFERENCE=true ;;
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

# Read the event IDs from the CSV file
csv_file="../event_status.csv"
gw_ids=$(awk -F, 'NR>1 {print $1}' $csv_file)

# Loop over each GW event ID
for gw_id in $gw_ids
do
  # Define the result directory path
  result_dir="$OUTDIR/$gw_id"
  
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

  if [ "$USE_NODE_PREFERENCE" = true ]; then
    # First try to find available c node
    AVAILABLE_NODE=$(sinfo -h -t idle -o "%n" | grep '^c' | head -n1)
    
    # If no c node, try to find available a node. Skip a2 nodes
    if [ -z "$AVAILABLE_NODE" ]; then
      AVAILABLE_NODE=$(sinfo -h -t idle -o "%n" | grep '^a' | grep -v '^a2' | head -n1)
    fi
    
    # Submit the job to SLURM only if preferred node is available
    if [ -n "$AVAILABLE_NODE" ]; then
      sbatch --nodelist=$AVAILABLE_NODE $new_script
      echo "Submitted job for $gw_id on node $AVAILABLE_NODE"
      # Wait for 5 seconds before submitting the next job
      sleep 5
    else
      echo "Skipping job for $gw_id - no a or c nodes available"
    fi
  else
    # Submit without node preference
    sbatch $new_script
    echo "Submitted job for $gw_id without node preference"
  fi
done
