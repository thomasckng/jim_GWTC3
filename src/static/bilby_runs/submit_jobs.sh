#!/bin/bash

# Activate the conda environment
source ~/.bashrc
conda activate igwn

# Read the event IDs from the CSV file
csv_file="../event_status.csv"
gw_ids=$(awk -F, 'NR>1 {print $1}' $csv_file)

# Loop over each GW event ID
for gw_id in $gw_ids
do
  # Define the result directory path
  result_dir="outdir"
  
  # Check if the result directory contains a subdirectory named as the GW ID
  if [ -d "$result_dir/$gw_id" ]; then
    echo "Result already exists for $gw_id, skipping submission."
    continue
  fi

  # Submit the job and wait for it to finish
  bilby_pipe config/${gw_id}_config.ini --submit
  wait
  
  echo "Submitted job for $gw_id"
done

# Deactivate the conda environment
conda deactivate
