#!/bin/bash
# ==============================================================================
# SCRIPT: Code/run_bayes_opt.sh
# DESCRIPCIÓN: Lanzamiento de la Optimización Bayesiana clásica en el CESGA FT3.
# ==============================================================================
#SBATCH -J tfg_bayes_opt
#SBATCH -p medium
#SBATCH -t 16:00:00
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH -o logs/bayes_opt_%j.out
#SBATCH -e logs/bayes_opt_%j.err

set -e

# Convertir el directorio home no escribible a la partición Store escribible automáticamente
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

# Asegurar dependencias del solver
pip install -q scikit-optimize pandas numpy

echo "=== [SLURM] Ejecutando solve_bayes_opt.py con 5 semillas en paralelo ==="
python src/solve_bayes_opt.py
echo "=== [SLURM] Finalizado solve_bayes_opt.py ==="
