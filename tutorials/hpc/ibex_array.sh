#!/bin/bash
# =============================================================================
#  BARISTA on Ibex — job array: one sample (sub-folder) per array task,
#  each task gets a full node's worth of cores.
#  layout:   samples/sample_A/*.csv   samples/sample_B/*.xlsx  ...
#  submit:   N=$(ls -d samples/*/ | wc -l); sbatch --array=0-$((N-1)) ibex_array.sh samples
# =============================================================================
#SBATCH --job-name=barista-array
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=barista_%A_%a.out

set -euo pipefail
source /home/shuklagn/anaconda3/etc/profile.d/conda.sh
conda activate barista
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export BARISTA_WORKERS=${SLURM_CPUS_PER_TASK}

ROOT=${1:?"usage: sbatch --array=0-N ibex_array.sh <samples root>"}
mapfile -t SAMPLES < <(ls -d "${ROOT}"/*/)
S=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
NAME=$(basename "$S")
# optional per-sample device file:  samples/sample_A/device.yaml
DEVARG=""
[[ -f "${S}/device.yaml" ]] && DEVARG="--device-file ${S}/device.yaml"

srun --cpu-bind=none barista analyze "$S" -o "results/${NAME}" --workers "${SLURM_CPUS_PER_TASK}" ${DEVARG}
