#!/bin/bash

# Define usage
usage() {
    echo "Usage: $0 [-o suffix] [-e 0|1|2] [-r 0|1|2] [-p] [-d] [-t]"
    echo "  -o suffix     Output directory suffix (creates outdir_SUFFIX, default: no suffix)"
    echo "  -e 0|1|2      Exclude nodes: 0=a2 (default), 1=a2,b[1-8], 2=a[1-9],b[1-8]"
    echo "  -r 0|1|2      Relative binning: 0=normal likelihood (default), 1=relative binning with fixed reference parameters, 2=relative binning with optimized reference parameters"
    echo "  -p            Use bilby PSD instead of computing from GWOSC data"
    echo "  -d            Use bilby frequency domain strain data instead of loading from GWOSC"
    echo "  -t            Test mode: only run GW150914"
    exit 1
}

OUTDIR_SUFFIX=""
EXCLUDE_NODES=0
RELATIVE_BINNING=0
USE_BILBY_PSD=0
USE_BILBY_DATA=0
TEST_MODE=0

while getopts "o:e:r:pdt" opt; do
    case $opt in
        o) OUTDIR_SUFFIX=$OPTARG ;;
        e) EXCLUDE_NODES=$OPTARG ;;
        r) RELATIVE_BINNING=$OPTARG ;;
        p) USE_BILBY_PSD=1 ;;
        d) USE_BILBY_DATA=1 ;;
        t) TEST_MODE=1 ;;
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

if [ -z "$OUTDIR_SUFFIX" ]; then
    OUTDIR="outdir"
else
    OUTDIR="outdir_$OUTDIR_SUFFIX"
fi

template_file="template.sh"

mkdir -p slurm_scripts "$OUTDIR"

csv_file="../event_status.csv"
if [ "$TEST_MODE" -eq 1 ]; then
    gw_ids="GW150914"
else
    gw_ids=$(awk -F, 'NR>1 {print $1}' "$csv_file")
fi

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
  
  if [ "$USE_BILBY_PSD" -eq 1 ]; then
    result_dir="$result_dir --use-bilby-psd"
  fi
  
  if [ "$USE_BILBY_DATA" -eq 1 ]; then
    result_dir="$result_dir --use-bilby-data"
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
