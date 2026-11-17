#!/bin/bash
# =============================================================================
#  BARISTA on Shaheen III — job array, one sample folder per array task (one node each)
#  submit:  N=$(ls -d samples/*/ | wc -l); sbatch --array=0-$((N-1)) shaheen_array.sh samples
# =============================================================================
#SBATCH --job-name=barista-array
#SBATCH --account=k1234             # <-- your Shaheen project ID
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=barista_%A_%a.out

set -euo pipefail
source /scratch/${USER}/miniforge3/etc/profile.d/conda.sh
conda activate barista
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export BARISTA_WORKERS=${SLURM_CPUS_PER_TASK}
export MPLCONFIGDIR=/scratch/${USER}/.cache/matplotlib; mkdir -p "${MPLCONFIGDIR}"

ROOT=${1:?"usage: sbatch --array=0-N shaheen_array.sh <samples root>"}
mapfile -t SAMPLES < <(ls -d "${ROOT}"/*/)
S=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
NAME=$(basename "$S")
DEVARG=""
[[ -f "${S}/device.yaml" ]] && DEVARG="--device-file ${S}/device.yaml"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=none \
     barista analyze "$S" -o "results/${NAME}" --workers "${SLURM_CPUS_PER_TASK}" ${DEVARG}
