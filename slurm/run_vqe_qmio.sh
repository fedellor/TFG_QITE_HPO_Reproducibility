#!/bin/bash
# ==============================================================================
# SCRIPT: Code/run_vqe_qmio.sh
# DESCRIPCIÓN: Ejecución física del algoritmo cuántico variacional VQE en la QPU real de Qmio.
#              Utiliza el hardware físico a través de QmioBackend y guarda resultados
#              en resultados_vqe_qmio.csv.
# ==============================================================================
#SBATCH -J tfg_vqe_qmio
#SBATCH -p qpu
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH -o logs/vqe_qmio_%j.out
#SBATCH -e logs/vqe_qmio_%j.err

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
module load qmio-run
module load qmio-tools

VENV_PATH="/mnt/netapp2/Store_uni/home/usc/cursos/curso1276/tfg_env_qmio"
if [ ! -d "$VENV_PATH" ]; then
    echo "=== Creando entorno virtual para Qmio con Python 3.9.9 ==="
    python3 -m venv "$VENV_PATH"
fi

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: No se encuentra el entorno virtual en $VENV_PATH."
    exit 1
fi

pip install -q "qiskit<1.1" "qiskit-algorithms==0.3.0" qiskit-optimization qiskit-aer pandas numpy

echo "=== [SLURM] VQE EN REAL QPU QMIO — 5 semillas ==="
export NOISE_MODE=qmio
export USE_REAL_QPU=1
export QMIO_CALIBRATIONS=/opt/cesga/qmio/hpc/calibrations
export PYTHONIOENCODING=utf-8
python src/solve_vqe.py
echo "=== [SLURM] Finalizado VQE en la QPU de Qmio. Resultado: resultados_vqe_qmio.csv ==="
