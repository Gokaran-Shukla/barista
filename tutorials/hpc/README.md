# Running BARISTA on Ibex and Shaheen III

BARISTA runs on **a single node** and uses every core you request. The 13 global mechanism fits,
the cross-validation refits, the two-channel screening and refinement, and the series-resistance
test are all spread over a process pool:

* The number of workers is set by, in order of priority: `--workers`, `BARISTA_WORKERS`,
  `SLURM_CPUS_PER_TASK`, and then the CPUs in the job's cpuset.
* Each worker is pinned to one BLAS/OpenMP thread, so *N* workers occupy exactly *N* cores.
* `--workers 1` runs serially. The results are identical to the parallel run.

Measured speed-up: a 480-point, 4-temperature × 3-thickness dataset took 68 s on 1 core and 34 s
on 2 cores. Scaling is near-linear until the number of cores exceeds the number of independent
fits (≈ 13 + 5 + number of two-channel pairs). Beyond that, run several samples at once with the
job-array scripts.

## One-time installation

### Ibex
```bash
source /home/$USER/anaconda3/etc/profile.d/conda.sh
conda create -n barista python=3.11 -y && conda activate barista
git clone https://github.com/Gokaran-Shukla/barista.git && cd barista
pip install -e ".[all]"
barista --version
```

### Shaheen III (install on /scratch — compute nodes do not mount /home)
```bash
cd /scratch/$USER
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /scratch/$USER/miniforge3
source /scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda create -n barista python=3.11 -y && conda activate barista
git clone https://github.com/Gokaran-Shukla/barista.git && cd barista
pip install -e ".[all]"
```

## Submitting

| script | what it does |
|---|---|
| `ibex_submit.sh` | one dataset, one Ibex node (40 cores by default) |
| `ibex_array.sh` | one sub-folder per array task, e.g. a whole sample library |
| `shaheen_submit.sh` | one dataset, one Shaheen III CPU node (192 cores) |
| `shaheen_array.sh` | job array on Shaheen III |

```bash
# Ibex: a single dataset
sbatch tutorials/hpc/ibex_submit.sh data/AlOx_series --area 100 --n-opt 1.65

# Ibex: every sample folder in samples/
N=$(ls -d samples/*/ | wc -l); sbatch --array=0-$((N-1)) tutorials/hpc/ibex_array.sh samples

# Shaheen III (set --account first)
sbatch tutorials/hpc/shaheen_submit.sh /scratch/$USER/data/HfO2 --material high_k_oxide
```

Optional device information per sample goes in `samples/<name>/device.yaml`:
```yaml
area_um2: 100
n_opt: 2.1
trap_density_cm3: 1.0e18
material_class: high_k_oxide
fix: [trap_density_cm3]
```

Monitor jobs with `squeue -u $USER`; the summary is in `barista_<jobid>.out`; open
`<output>/report.html` for the full report.
