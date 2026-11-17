#!/bin/bash
# =============================================================================
#  BARISTA on KAUST Ibex — one node, all requested cores
#  submit:  sbatch ibex_submit.sh  <data folder or files>  [extra barista options]
#  example: sbatch ibex_submit.sh ./AlOx_series --area 100 --n-opt 1.65
# =============================================================================
#SBATCH --job-name=barista
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40          # BARISTA uses every core given here (one process per core)
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=barista_%j.out
#SBATCH --error=barista_%j.err
##SBATCH --constraint=intel         # optional: pin a CPU family
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=gokaran.shukla@kaust.edu.sa

set -euo pipefail

# ---- environment (conda env created once with:  conda create -n barista python=3.11 ; pip install -e ".[all]")
source /home/shuklagn/anaconda3/etc/profile.d/conda.sh
conda activate barista

# ---- one BLAS thread per worker; BARISTA spawns SLURM_CPUS_PER_TASK workers
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BARISTA_WORKERS=${SLURM_CPUS_PER_TASK}
export MPLBACKEND=Agg

INPUT=${1:?"usage: sbatch ibex_submit.sh <data> [barista options]"}
shift || true
OUT=barista_${SLURM_JOB_ID}_$(basename "${INPUT%.*}")

echo "Node: $(hostname)  cores: ${SLURM_CPUS_PER_TASK}  start: $(date)"
srun --cpu-bind=none barista analyze "${INPUT}" -o "${OUT}" --workers "${SLURM_CPUS_PER_TASK}" \
     --formats pdf,png,svg,eps -v "$@"
echo "Finished: $(date)  -> ${OUT}/report.html"
