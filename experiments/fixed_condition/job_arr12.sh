#!/bin/bash
#SBATCH --job-name=datamatters
#SBATCH --partition=cpu-5h
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out
#SBATCH --mem=2G

# all train/val size combos
COMBOS=( "1:0" "3:0" "5:0" "10:0" "15:0" "20:0" "25:0" "30:0" "35:0" "40:0" "45:0" "49:0" )
TOTAL=${#COMBOS[@]}

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "This script expects to run as a Slurm array task. Submit with --array=0-$((TOTAL-1))"
  exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 1
fi

# pick the pair for this task
pair=${COMBOS[$SLURM_ARRAY_TASK_ID]}
n_train=${pair%%:*}
n_val=${pair##*:}
n_test=10

# first two positional args = data_path; remaining args forwarded as seed list
DATA_PATH="$1"
shift
SEEDS="$@"   # e.g. call: job_array.sh /datapath/data.csv 1 2 3 4 5

echo "task $SLURM_ARRAY_TASK_ID: train=$n_train val=$n_val test=$n_test seeds=[$SEEDS]"

apptainer run --nv /home/n.gebauer/projects/datamatters/datamatters_container.sif \
  python /home/n.gebauer/projects/datamatters/code/src/ocm_xgb.py \
  --data_path "$DATA_PATH" --seeds $SEEDS \
  --n_train_catalysts "$n_train" --n_test_catalysts "$n_test" --split_strategy catalyst --n_folds 5 --cross_val_params 50 \
  --feature_sets base+atom_numbers+support base+descriptors all one_hot invariant --compute_shap
