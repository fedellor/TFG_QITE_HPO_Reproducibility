# Orquestación HPC (`slurm/`)

Esta carpeta contiene los scripts Bash diseñados para interactuar con el gestor de colas Slurm en el supercomputador FinisTerrae III (FT3) del CESGA, permitiendo automatizar los entrenamientos paralelos, barridos y el enrutamiento físico a la QPU cuántica Qmio.

## Archivos de Orquestación

*   **`setup_env.sh`**: Script de aprovisionamiento que automatiza la creación de un entorno virtual Python (`venv`) en la partición `/Store` del CESGA, instalando PyTorch con soporte CUDA y librerías cuánticas de Qiskit compatibles con el hardware físico.
*   **`launch_hpo.sh`**: Script de tipo array de Slurm (tareas 1-100) que lanza de forma concurrente el muestreo del pool HPO clásico (300 ejecuciones totales repartidas entre 5 semillas) en nodos acelerados con GPUs NVIDIA A100.
*   **`run_bayes_opt.sh`**: Envío de Slurm para ejecutar la optimización clásica basada en procesos gaussianos sobre las 5 semillas del TFG.
*   **`run_vqe.sh`**: Script para ejecutar las optimizaciones VQE simuladas en el supercomputador FT3.
*   **`run_qite.sh`**: Script para ejecutar las simulaciones cuánticas VarQITE (evolución en tiempo imaginario) clásica.
*   **`run_vqe_qmio.sh`**: Script de enrutamiento cuántico que configura los accesos de red y ejecuta VQE conectándose directamente al hardware superconductor real de la QPU Qmio de 32 qubits.
*   **`run_qite_qmio.sh`**: Script de enrutamiento cuántico que ejecuta el algoritmo VarQITE en la QPU física de Qmio aprovechando el esquema de evolución paralela de circuitos.
