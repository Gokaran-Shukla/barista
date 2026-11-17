#!/bin/bash
# =============================================================================
#  BARISTA on KAUST Shaheen III (CPU partition) — one full node
#  Shaheen III CPU nodes: 2 x AMD EPYC 9654 (Genoa), 192 cores.
#  Run from /scratch: compute nodes do not see /home.
#  submit:  sbatch shaheen_submit.sh <data folder or files> [extra barista options]
# =============================================================================
#SBATCH --job-name=barista
#SBATCH --account=k1234             # <-- your Shaheen project ID
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192         # all physical cores of the node
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=barista_%j.out
#SBATCH --error=barista_%j.err

set -euo pipefail

# ---- environment: a conda/miniforge install on /scratch (see tutorials/hpc/README.md)
source /scratch/${USER}/miniforge3/etc/profile.d/conda.sh
conda activate barista

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export BARISTA_WORKERS=${SLURM_CPUS_PER_TASK}
export MPLCONFIGDIR=/scratch/${USER}/.cache/matplotlib   # writable cache on compute nodes
mkdir -p "${MPLCONFIGDIR}"

INPUT=${1:?"usage: sbatch shaheen_submit.sh <data> [barista options]"}
shift || true
OUT=barista_${SLURM_JOB_ID}_$(basename "${INPUT%.*}")

echo "Node: $(hostname)  cores: ${SLURM_CPUS_PER_TASK}  start: $(date)"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=none \
     barista analyze "${INPUT}" -o "${OUT}" --workers "${SLURM_CPUS_PER_TASK}" --formats pdf,png,svg,eps -v "$@"
echo "Finished: $(date)  -> ${OUT}/report.html"
