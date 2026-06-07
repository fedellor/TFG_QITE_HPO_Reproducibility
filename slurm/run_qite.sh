#!/bin/bash
# ==============================================================================
# SCRIPT: Code/run_qite.sh
# DESCRIPCIÓN: Ejecución clásica del barrido de ruido VarQITE
# ==============================================================================
#SBATCH -J qite
#SBATCH -p medium
#SBATCH -t 16:00:00
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH -o logs/qite_sweep_qmio_%j.out
#SBATCH -e logs/qite_sweep_qmio_%j.err

set -e

# Convertir el directorio home no escribible a la partición Store automáticamente
if [[ "$SLURM_SUBMIT_DIR" == /home/usc/cursos/curso1276* ]]; then
    export WORK_DIR="${SLURM_SUBMIT_DIR/\/home\/usc\/cursos\/curso1276/\/mnt/netapp2/Store_uni/home/usc/cursos/curso1276}"
else
    export WORK_DIR="$SLURM_SUBMIT_DIR"
fi
cd "$WORK_DIR"
mkdir -p logs

module purge
module load python/3.9.9
module load gcc/12.3.0

VENV_PATH="/mnt/netapp2/Store_uni/home/usc/cursos/curso1276/tfg_env"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: No se encuentra el entorno virtual en $VENV_PATH."
    exit 1
fi

export NOISE_MODE=sweep
export USE_REAL_QPU=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

python src/solve_qite.py
echo "=== [SLURM] Finalizado VarQITE Sweep. Resultados: resultados_qite_sweep.csv ==="
