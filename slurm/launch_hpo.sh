#!/bin/bash
# ==============================================================================
# SCRIPT: Code/launch_hpo.sh
# DESCRIPCIÓN: Planificador SLURM Job Array para entrenamiento paralelo en GPU en CESGA FT3.
# ==============================================================================

# Directivas SLURM de alto rendimiento (Aceleración por GPU)
# ------------------------------------------------------------------------------
#SBATCH -J tfg_hpo_transformer        # Nombre identificativo del Job
#SBATCH -p short                       # Cola 'short' (6 horas de límite)
#SBATCH -t 04:00:00                    # Tiempo límite solicitado (4 horas)
#SBATCH -c 32                          # 32 núcleos CPU físicos asignados por tarea 
#SBATCH --mem=64G                      # 64 Gigabytes de RAM asignados al host por tarea
#SBATCH --gres=gpu:a100:1              # Solicitar explícitamente 1 GPU NVIDIA A100 por tarea
#SBATCH --array=1-50                   # Lanzar 50 tareas paralelas
#SBATCH -o logs/hpo_%A_%a.out          # Canal de salida estándar redirigido a logs/
#SBATCH -e logs/hpo_%A_%a.err          # Canal de errores redirigido a logs/

set -e # Salir en caso de error

# Asegurar que el script se ejecute tomando como referencia su propia ubicación (Code/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Crear el directorio de registros logs si no existe en el sistema de archivos
mkdir -p logs

# Carga de módulos requeridos y activación del entorno virtual de Python con CUDA
echo "=== [SLURM GPU JOB ARRAY] Iniciando tarea $SLURM_ARRAY_TASK_ID en el nodo $SLURMD_NODENAME ==="
module purge
module load python/3.10.8
module load gcc/12.3.0

# Activar el entorno virtual (ruta dinámica del CESGA)
VENV_PATH="${store:-$HOME}/tfg_env"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: No se encuentra el entorno virtual en $VENV_PATH. Ejecute primero 'setup_env.sh'."
    exit 1
fi

# Leer el registro de hiperparámetros correspondiente del archivo pool_jobs.csv
CSV_FILE="pool_jobs.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: El archivo de pool '$CSV_FILE' no existe. Ejecute 'python sampler_hpo.py' primero."
    exit 1
fi

# Cada tarea de la 1 a la 50 procesará exactamente 6 configuraciones de forma secuencial
# Tarea 1: indices 1, 51, 101, 151, 201, 251
# Tarea 2: indices 2, 52, 102, 152, 202, 252
# ...
for OFFSET in 0 50 100 150 200 250; do
    CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID + OFFSET))
    echo "=========================================================================="
    echo "=== PROCESANDO CONFIGURACIÓN HPO ÍNDICE: $CONFIG_INDEX / 300 ==="
    echo "=========================================================================="

    # Dado que la fila 1 es la cabecera, la fila $CONFIG_INDEX + 1 es el registro de hiperparámetros
    ROW_INDEX=$((CONFIG_INDEX + 1))
    LINE_CONTENT=$(sed -n "${ROW_INDEX}p" "$CSV_FILE")

    if [ -z "$LINE_CONTENT" ]; then
        echo "WARNING: No se pudo obtener la línea $ROW_INDEX de '$CSV_FILE' (Índice fuera de rango). Saltando..."
        continue
    fi

    # Desestructurar la línea leída para extraer los hiperparámetros individuales
    IFS=',' read -r INDEX BATCH_SIZE LEARNING_RATE EMB_SIZE ATTN_HEADS ENC_LAYERS DEC_LAYERS <<< "$LINE_CONTENT"

    echo "Fila leída ($ROW_INDEX): INDEX=$INDEX | BS=$BATCH_SIZE | LR=$LEARNING_RATE | EMB=$EMB_SIZE | HEADS=$ATTN_HEADS | ENC=$ENC_LAYERS | DEC=$DEC_LAYERS"

    # Configurar variables de optimización de hilos para soporte CUDA y prevención de cuellos de botella
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

    # Ejecutar el entrenamiento con validación cruzada de 5 folds en GPU
    python src/train_hpo.py \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --emb_size "$EMB_SIZE" \
        --attn_heads "$ATTN_HEADS" \
        --enc_layers "$ENC_LAYERS" \
        --dec_layers "$DEC_LAYERS" \
        --epochs 50 \
        --log_name "env_permit" \
        --results_csv "resultados_hpo.csv" \
        --job_id "$CONFIG_INDEX"
done

echo "=== [SLURM GPU JOB ARRAY] Tarea $SLURM_ARRAY_TASK_ID completada con éxito en el nodo $SLURMD_NODENAME ==="

