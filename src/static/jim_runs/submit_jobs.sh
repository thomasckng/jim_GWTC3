#!/bin/bash

# Define usage
usage() {
    echo "Usage: $0 [-o outdir] [-e 0|1|2] [-r 0|1|2]"
    echo "  -o outdir     Output directory (default: outdir)"
    echo "  -e 0|1|2      Exclude nodes: 0=a2 (default), 1=a2,b[1-8], 2=a[1-9],b[1-8]"
    echo "  -r 0|1|2      Relative binning: 0=normal likelihood (default), 1=relative binning with fixed reference parameters, 2=relative binning with optimized reference parameters"
    exit 1
}

OUTDIR=""
EXCLUDE_NODES=0
RELATIVE_BINNING=0

while getopts "o:e:r:" opt; do
    case $opt in
        o) OUTDIR=$OPTARG ;;
        e) EXCLUDE_NODES=$OPTARG ;;
        r) RELATIVE_BINNING=$OPTARG ;;
        ?) usage ;;
    esac
    # Check for missing argument
    if [[ $OPTARG == -* ]]; then
        usage
    fi
done

# Validate EXCLUDE_NODES
if ! [[ "$EXCLUDE_NODES" =~ ^[0-2]$ ]]; then
    echo "Invalid value for -e. Use 0, 1, or 2."
    usage
fi
# Validate RELATIVE_BINNING
if ! [[ "$RELATIVE_BINNING" =~ ^[0-2]$ ]]; then
    echo "Invalid value for -r. Use 0, 1, or 2."
    usage
fi

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

  if [ "$EXCLUDE_NODES" -eq 0 ]; then
    sbatch --exclude=a2 "$new_script"
  elif [ "$EXCLUDE_NODES" -eq 1 ]; then
    sbatch --exclude=a2,b[1-8] "$new_script"
  elif [ "$EXCLUDE_NODES" -eq 2 ]; then
    sbatch --exclude=a[1-9],b[1-8] "$new_script"
  fi

  echo "Submitted job for $JOB_NAME"
done
