#!/bin/bash

# Define usage
usage() {
    echo "Usage: $0 [-o outdir] [-e] [-r] (use -o to specify output directory, -e to exclude nodes b[1-8], -r to use relative binning)"
    exit 1
}

OUTDIR=""
EXCLUDE_NODES=false
RELATIVE_BINNING=0

while getopts "o:er:" opt; do
    case $opt in
        o) OUTDIR=$OPTARG ;;
        e) EXCLUDE_NODES=true ;;
        r) RELATIVE_BINNING=$OPTARG ;;
        ?) usage ;;
    esac
done

if [ -z "$OUTDIR" ]; then
    OUTDIR="outdir"
fi

template_file="template.sh"

mkdir -p slurm_scripts "$OUTDIR"

csv_file="../event_status.csv"
gw_ids=$(awk -F, 'NR>1 {print $1}' "$csv_file")

for gw_id in $gw_ids; do
  if [ "$RELATIVE_BINNING" -eq 1 ]; then
    LABEL="_rb_fixed"
  elif [ "$RELATIVE_BINNING" -eq 2 ]; then
    LABEL="_rb_free"
  else
    LABEL=""
  fi

  JOB_NAME="${gw_id}${LABEL}"
  result_dir="$OUTDIR"
  if [ "$RELATIVE_BINNING" -eq 1 ]; then
    result_dir="$result_dir -r 1"
  elif [ "$RELATIVE_BINNING" -eq 2 ]; then
    result_dir="$result_dir -r 2"
  fi

  # Skip only if the per-event output dir contains samples.npz
  event_dir="$OUTDIR/$gw_id"
  if [ -f "$event_dir/samples.npz" ]; then
    echo "Skipping $gw_id: $event_dir/samples.npz exists"
    continue
  fi

  # Create the per-event output directory
  mkdir -p "$event_dir"

  new_script="slurm_scripts/submit_${JOB_NAME}.sh"
  cp "$template_file" "$new_script"

  sed -i "s/{{{JOB_NAME}}}/$JOB_NAME/g" "$new_script"
  sed -i "s/{{{GW_ID}}}/$gw_id/g"       "$new_script"
  sed -i "s/{{{OUTDIR}}}/$OUTDIR/g"     "$new_script"
  sed -i "s#default#$result_dir#g"      "$new_script"

  chmod +x "$new_script"

  if [ "$EXCLUDE_NODES" = true ]; then
    sbatch --exclude=a2,b[1-8] "$new_script"
  else
    sbatch --exclude=a2 "$new_script"
  fi

  echo "Submitted job for $JOB_NAME"
done
