#!/bin/bash
# ==============================================================================
# SCRIPT: Code/run_vqe.sh
# DESCRIPCIÓN: Lanzamiento de la simulación cuántica VQE en el CESGA FT3.
# ==============================================================================
#SBATCH -J tfg_vqe_sweep
#SBATCH -p medium
#SBATCH -t 16:00:00
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH -o logs/vqe_sweep_%j.out
#SBATCH -e logs/vqe_sweep_%j.err

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
module load python/3.10.8
module load gcc/12.3.0

if [ -d "/mnt/netapp2/Store_uni/home/usc/cursos/curso1276/tfg_env" ]; then
    VENV_PATH="/mnt/netapp2/Store_uni/home/usc/cursos/curso1276/tfg_env"
elif [ -d "/mnt/netapp2/Store_uni//home/usc/cursos/curso1276/tfg_env" ]; then
    VENV_PATH="/mnt/netapp2/Store_uni//home/usc/cursos/curso1276/tfg_env"
elif [ -n "$STORE" ] && [ -d "$STORE/tfg_env" ]; then
    VENV_PATH="$STORE/tfg_env"
else
    VENV_PATH="$HOME/tfg_env"
fi

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: No se encuentra el entorno virtual en $VENV_PATH."
    exit 1
fi

# Asegurar dependencias del solver con versiones alineadas compatibles
pip install -q "qiskit<1.1" "qiskit-algorithms==0.3.0" qiskit-optimization qiskit-aer pandas numpy

echo "=== [SLURM] Ejecutando solve_vqe.py en modo BARRIDO DE RUIDO (sweep) ==="
export NOISE_MODE=sweep
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
python src/solve_vqe.py
echo "=== [SLURM] Finalizado solve_vqe.py. Resultados guardados en la carpeta resultados/ ==="
