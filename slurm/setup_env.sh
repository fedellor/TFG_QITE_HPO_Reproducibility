#!/bin/bash
# ==============================================================================
# SCRIPT: Code/setup_env.sh
# DESCRIPCIÓN: Inicialización del entorno virtual aislado en FinisTerrae III (CESGA).
# ==============================================================================

set -e # Terminar inmediatamente si ocurre algún error

echo "=== [1/4] Limpiando el entorno actual (Module Purge) ==="
module purge

echo "=== [2/4] Cargando módulos nativos requeridos en FT3 (Con soporte CUDA) ==="
# Cargamos Python, GCC de compilación nativa y el módulo CUDA del supercomputador para GPU
module load python/3.10.8
module load gcc/12.3.0

# Definir ruta dinámica del entorno virtual. Usa la partición de almacenamiento rápido $store si existe, o el $HOME en su defecto.
VENV_PATH="${store:-$HOME}/tfg_env"
echo "=== [3/4] Creando el entorno virtual en $VENV_PATH ==="
if [ -d "$VENV_PATH" ]; then
    echo "El entorno virtual ya existe. Se procederá a su reinstalación/actualización."
else
    python3 -m venv "$VENV_PATH"
fi

echo "=== [4/4] Activando el entorno virtual e instalando dependencias ==="
source "$VENV_PATH/bin/activate"

# Aseguramos la última versión de pip, setuptools y wheel
pip install --upgrade pip setuptools wheel

# Instalamos el árbol exacto de dependencias de Deep Learning con soporte GPU (PyTorch)
echo "Instalando paquetes científicos y de minería de procesos..."
pip install \
    torch==2.2.1 \
    numpy==1.26.4 \
    pandas==2.2.1 \
    scikit-learn==1.4.1.post1 \
    pyyaml==6.0.1 \
    tqdm==4.66.2 \
    pm4py==2.7.8.2 \
    scipy==1.12.0 \
    matplotlib==3.8.3 \
    seaborn==0.13.2

echo "=============================================================================="
# Verificamos la carga y compatibilidad nativa con GPUs (CUDA)
python -c "import torch; print('PyTorch cargado correctamente. CUDA disponible:', torch.cuda.is_available()); print('Dispositivo actual:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Ninguno')"
python -c "import pm4py; print('PM4Py instalado correctamente.')"
echo "=== ¡ENTORNO VIRTUAL CONFIGURADO E INSTALADO CON ÉXITO EN $VENV_PATH! ==="
echo "=============================================================================="
