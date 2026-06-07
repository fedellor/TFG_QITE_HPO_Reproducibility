# Estudio de Mecanismos Cuánticos para la Optimización de Hiperparámetros en Redes Neuronales Profundas

Este repositorio contiene el código fuente, conjuntos de datos, solucionadores clásicos/cuánticos y scripts de supercomputación desarrollados para el Trabajo de Fin de Grado (TFG): **"Estudio de mecanismos cuánticos para la optimización de hiperparámetros en redes neuronales profundas"**.

El problema de optimización de hiperparámetros (HPO) se modela mediante una formulación matemática de optimización binaria sin restricciones cuadráticas (**QUBO**) y se resuelve utilizando resolvedores cuánticos variacionales de vanguardia (**VQE** y **VarQITE**) y optimización clásica (**Optimización Bayesiana**), ejecutados en la infraestructura del **CESGA** (Supercomputador FinisTerrae III y el computador cuántico Qmio).

---

## Arquitectura y Estructura del Repositorio

El repositorio se encuentra estructurado de forma limpia y modular para garantizar la reproducibilidad y el rigor académico:

```
├── data/                           # Datasets de entrada y matrices intermedias del QUBO
│   ├── resultados_hpo.csv          # Pool HPO: 300+ ejecuciones reales del Transformer (5-Fold CV, 50 épocas)
│   ├── matriz_qubo.csv             # Matriz Q binaria de acoplamientos (30x30)
│   └── metricas_qubo.json          # Parámetros del fit Lasso L1 LARS CV y multiplicador de Lagrange λ
├── resultados/                     # Resultados máster unificados de los solvers e imágenes
│   ├── resultados_bayes_opt.csv    # Resultados de Optimización Bayesiana clásica (5 semillas)
│   ├── resultados_vqe_sweep.csv    # Resultados de simulación clásica VQE con barrido de ruido (FT3)
│   ├── resultados_qite_sweep.csv   # Resultados de simulación clásica VarQITE con barrido de ruido (FT3)
│   ├── resultados_vqe_qmio.csv     # Resultados de VQE en el hardware cuántico físico real de Qmio
│   ├── resultados_qite_qmio.csv    # Resultados de VarQITE en el hardware cuántico físico real de Qmio
│   ├── resultados_soluciones_reales.csv # Comparativa consolidada de soluciones y métricas reales del Transformer
│   ├── comparativa_tiempos.png     # Gráfico comparativo de tiempos (Simulación vs QPU Real)
│   ├── comparativa_energia.png     # Gráfico comparativo de energía media del QUBO
│   └── comparativa_factibilidad.png # Gráfico comparativo de factibilidad (violaciones One-Hot)
├── docs/                           # Memorias, manuales de diseño y formulación matemática
│   ├── TFG_Master_Report_HPO.md    # Reporte completo del TFG y análisis comparativo HPO
│   ├── QUBO_MultiObjective_Formulation.md # Formulación matemática y mapeo One-Hot formal
│   ├── Formula QUBO.png            # Render gráfico del Hamiltoniano de Ising
│   └── GrIA_TFG_Modelo_A_Memoria.pdf # Memoria académica oficial en PDF
├── slurm/                          # Scripts de envío de tareas al clúster (Slurm)
│   ├── setup_env.sh                # Despliegue del entorno virtual (tfg_env) en el clúster
│   ├── launch_hpo.sh               # Envío del Job Array paralelo (1-50) para el pool HPO inicial
│   ├── run_bayes_opt.sh            # Trabajo Slurm para resolvedor bayesiano clásico en FT3
│   ├── run_vqe.sh                  # Trabajo Slurm para sweeps de simulación VQE con ruido
│   ├── run_vqe_qmio.sh             # Trabajo Slurm para VQE en la QPU real de Qmio
│   ├── run_qite_sweep_qmio.sh      # Trabajo Slurm para sweeps de simulación VarQITE en nodo de gran memoria
│   └── run_qite_qmio.sh            # Trabajo Slurm para VarQITE en la QPU real de Qmio
├── logs/                           # Historial de logs remotos y locales del clúster
├── dashboard.py                    # Cuadro de mando interactivo en Streamlit
├── dashboard_data.json             # Datos consolidados de soporte para el dashboard Streamlit
├── (Módulos principales del Transformer Neural Network en la raíz)
│   ├── config.py                   # Semillas aleatorias, hiperparámetros por defecto y directorios absolutos
│   ├── embeddings.py               # Capas de embeddings categóricos y numéricos del EventLog
│   ├── encoders_and_decoders.py    # Arquitectura EventTransformer (Mecanismo de atención multicanal)
│   ├── training.py                 # Funciones de entrenamiento, bucles de folds y cálculo de funciones de pérdida
│   ├── train_hpo.py                # Wrapper para la evaluación de 5-Fold Cross Validation del Transformer
│   └── utils.py                    # Procesamiento y vectorización de secuencias de logs de eventos
└── (Módulos de Solvers y Modelado en la raíz)
    ├── generate_qubo.py            # Modelador QUBO mediante regresión Lasso L1 LARS CV
    ├── solve_bayes_opt.py          # Solucionador clásico mediante Optimización Bayesiana Gaussiana
    ├── solve_vqe.py                # Solucionador cuántico VQE (Qiskit) con soporte de ruido depolarizante
    └── solve_qite.py               # Solucionador cuántico VarQITE con Euler de paso fijo
```

---

## El Pipeline de 5 Fases: De la Red Neuronal a la QPU Física

El flujo de trabajo es un **bucle clásico-cuántico-clásico cerrado** implementado en 5 fases operativas:

```mermaid
flowchart TD
    %% Estilos de Nodos
    classDef dataset fill:#1f77b4,stroke:#333,stroke-width:2px,color:#fff;
    classDef script fill:#2ca02c,stroke:#333,stroke-width:1px,color:#fff;
    classDef master fill:#ff7f0e,stroke:#333,stroke-width:2px,color:#fff;
    classDef cluster fill:#d62728,stroke:#333,stroke-width:1px,color:#fff;
    classDef step fill:#7f7f7f,stroke:#333,stroke-width:1px,color:#fff;

    %% Fase 1
    subgraph Fase_1 ["Fase 1: Datos y Pool Inicial HPO"]
        Raw["Event Log (env_permit)"] -->|Procesamiento de Logs| EventLog["utils.py & data_processors"]:::script
        EventLog -->|5-Fold CV (50 épocas por fold)| ClassicTrain["Entrenamiento Clásico (300+ combinaciones)"]:::step
        ClassicTrain -->|Registro de Acc, Loss y Tiempo| HPO_CSV[("data/resultados_hpo.csv")]:::dataset
    end

    %% Fase 2
    subgraph Fase_2 ["Fase 2: Modelado del Paisaje QUBO"]
        HPO_CSV -->|Muestras de Entrada| LassoReg["generate_qubo.py (Lasso L1 LARS CV)"]:::script
        LassoReg -->|Matriz Q (30x30 binaria)| QUBO_CSV[("data/matriz_qubo.csv")]:::dataset
        LassoReg -->|RMSE, λ óptimo y Coeficientes| Metrics_JSON[("data/metricas_qubo.json")]:::dataset
    end

    %% Fase 3
    subgraph Fase_3 ["Fase 3: Optimización (Solvers Cuánticos y Clásicos)"]
        QUBO_CSV & Metrics_JSON -->|Mapeo del Hamiltoniano de Ising| BayesSolver["solve_bayes_opt.py"]:::script
        QUBO_CSV & Metrics_JSON -->|Emulación de barrido de ruido en FT3| VQESolver["solve_vqe.py"]:::script
        QUBO_CSV & Metrics_JSON -->|Euler de paso fijo (10 steps)| QITESolver["solve_qite.py"]:::script

        BayesSolver -->|Resultados Clásicos| ResBayes[("resultados/resultados_bayes_opt.csv")]:::dataset
        VQESolver -->|Simulación de Ruido y QPU Real| ResVQE[("resultados/resultados_vqe_*.csv")]:::dataset
        QITESolver -->|Simulación de Ruido y QPU Real| ResQITE[("resultados/resultados_qite_*.csv")]:::dataset
    end

    %% Fase 4
    subgraph Fase_4 ["Fase 4: Bucle de Validación Real en GPU (Cerrado)"]
        ResBayes & ResVQE & ResQITE -->|Hiperparámetros Óptimos Encontrados| ValidOrch["Selección manual de soluciones y envío"]:::step
        HPO_CSV -->|Deduplicación manual (Evitar re-entrenamientos)| ValidOrch
        
        ValidOrch -->|Trabajos Slurm manuales (32 CPUs + GPU)| FT3Queue["Supercomputador FT3 (GPUs A100)"]:::cluster
        FT3Queue -->|Entrenamiento real (5-Fold, 50 épocas)| JobFolder["Fichero train_hpo.py manual"]:::step
        JobFolder -->|Métricas clásicas obtenidas| HPO_CSV
    end

    %% Fase 5
    subgraph Fase_5 ["Fase 5: Consolidación Comparativa y Visualización"]
        HPO_CSV -->|Sincronización manual de CSVs| SyncMetrics["Sincronización y copiado manual"]:::step
        SyncMetrics -->|Reemplaza -1.0 en masters por métricas reales| ResBayes & ResVQE & ResQITE
        
        ResBayes & ResVQE & ResQITE -->|Consolidación manual| CompReport["Integración en tabla general"]:::step
        CompReport -->|Genera el reporte máster consolidado| FinalCompare[("resultados/resultados_soluciones_reales.csv")]:::master
        HPO_CSV & FinalCompare -->|Visualización Visual| StreamlitDash["dashboard.py (Streamlit)"]:::script
    end
```

---

## Guía Metodológica Detallada Paso a Paso

### Paso 0: Despliegue del Entorno Virtual en FT3

Antes de comenzar, es necesario inicializar el entorno de Python en los nodos del CESGA:
```bash
# Otorgar permisos de ejecución e instalar dependencias en tfg_env
chmod +x slurm/setup_env.sh
./slurm/setup_env.sh
```

---

### Fase 1: Creación del Paisaje HPO Clásico (Los 300+ Experimentos)

Para optimizar hiperparámetros con algoritmos avanzados, primero se debe mapear o muestrear una parte representativa de la función de coste clásica de la red neuronal. 
1. **El Modelo**: Un **EventTransformer** (atención multicanal) diseñado para predecir la siguiente actividad, tiempo de ejecución y recursos en flujos de procesos.
2. **Evaluación de Rigor**: Se evalúa cada combinación de hiperparámetros mediante **validación cruzada de 5 pliegues (5-Fold Cross Validation)**.
3. **El Entrenamiento**: Para cada fold, el modelo se entrena durante **exactamente 50 épocas** (con optimizador Cosine Annealing / AdamW).
4. **Persistencia**: Se extraen la exactitud promedio (Concept Damerau-Levenshtein Similarity), pérdida (Cross Entropy) y tiempo total de cómputo en segundos y se guardan en el pool inicial `data/resultados_hpo.csv`.

*   **¿Cómo se ejecuta?** El script `slurm/launch_hpo.sh` utiliza la funcionalidad **Slurm Job Array** para programar hasta 50 combinaciones en paralelo en las GPUs NVIDIA A100 del FT3 para poblar la matriz:
    ```bash
    sbatch slurm/launch_hpo.sh
    ```

---

### Fase 2: Modelado del Paisaje y Construcción del QUBO

El conjunto de 300+ combinaciones discretas se transforma en una superficie matemática continua cuadrática de segundo orden mediante regresión estadística regularizada.
1. **Mapeo Binario**: Se seleccionan 6 hiperparámetros principales (Batch Size, Learning Rate, Embeddings, Attention Heads, Encoder Layers y Decoder Layers). Sus valores posibles se discretizan en **30 variables binarias** ($x_i \in \{0, 1\}$) con codificación **One-Hot**.
2. **Ajuste Estadístico**: El script `generate_qubo.py` utiliza la técnica de regresión **Lasso L1 LARS Cross-Validation** sobre las muestras de `data/resultados_hpo.csv` para ajustar los coeficientes cuadráticos de acoplamiento.
3. **Hamiltoniano de Ising**: Se construye el modelo energético de costo hamiltoniano, agregando penalizaciones matemáticas de **Lagrange ($\lambda$)** para castigar físicamente con alta energía (+20 unidades de energía por cúbit violado) a cualquier configuración cuántica que no cumpla con la restricción One-Hot de seleccionar un único valor por hiperparámetro.

*   **¿Cómo se ejecuta?**
    ```bash
    python generate_qubo.py
    ```
    *Salida*: `data/matriz_qubo.csv` (matriz de acoplamientos $Q$) y `data/metricas_qubo.json` (parámetros óptimos del fit y penalizaciones).

---

### Fase 3: Optimización del Paisaje (Solvers)

Con la matriz QUBO definida, los resolvedores buscan los vectores de espín binarios (estados fundamentales) que minimizan la energía del Hamiltoniano (que se corresponde directamente con los mejores hiperparámetros reales del Transformer).

#### 1. Optimización Bayesiana (Clásica de Referencia)
Mapea el QUBO de forma secuencial empleando procesos gaussianos a lo largo de 5 semillas aleatorias.
*   **Ejecución**: `python solve_bayes_opt.py` o mediante Slurm: `sbatch slurm/run_bayes_opt.sh`

#### 2. Variational Quantum Eigensolver (VQE)
Algoritmo cuántico variacional clásico que minimiza el valor esperado de la energía $\langle \psi(\theta) | H | \psi(\theta) \rangle$ optimizando los parámetros $\theta$ de un circuito parametrizado (ansatz).
*   **Ejecución (FT3 Sweeps de Ruido)**: Realiza simulaciones clásicas aceleradas de VQE sometiendo al circuito a barridos depolarizantes y errores de lectura (`ideal`, `low`, `medium` y `high` noise):
    ```bash
    sbatch slurm/run_vqe.sh
    ```

#### 3. Variational Quantum Imaginary Time Evolution (VarQITE)
Algoritmo cuántico avanzado que emula la evolución termodinámica no unitaria en tiempo imaginario $e^{-H\tau}$. Su ventaja teórica reside en que la penalización matemática de Lagrange $\lambda$ del One-Hot actúa aniquilando de forma exponencial ($e^{-E_n\tau}$) las componentes del estado cuántico que violen las restricciones, purificando la función de onda y garantizando una factibilidad absoluta.
*   **Ejecución (FT3 Sweeps de Ruido)**:
    ```bash
    sbatch slurm/run_qite.sh
    ```

#### 4. Ejecución en Hardware Cuántico Físico Real (QPU Qmio)
Para conectarse directamente a la QPU física real de Qmio, se lanzan los resolvedores divididos en **1 trabajo individual por semilla** para exprimir los recursos del nodo controlador cuántico evitando el límite general de CPU-minutos (`AssocGrpCPUMinutesLimit`):

*   **VarQITE en QPU (Exprimir RAM para QGT)**:
    VarQITE requiere alta memoria en el nodo para resolver el sistema variacional lineal de McLachlan de 30 cúbits. Se lanza con los recursos máximos del nodo de QPU (`-c 64`, `--mem=980G`) para 1 hora límite por semilla:
    ```bash
    sbatch -J qite_S1001 -p qpu -t 01:00:00 -c 64 --mem=980G --export=ALL,SINGLE_SEED=1001 -o logs/qite_qmio_seed_1001_%j.out -e logs/qite_qmio_seed_1001_%j.err slurm/run_qite_qmio.sh
    ```
*   **VQE en QPU (Optimización de Quota)**:
    Como VQE no requiere computar QGT ni derivadas en memoria clásica, se ejecuta de forma óptima con recursos mínimos (`-c 1`, `--mem=8G`), evitando el bloqueo de minutos de la asociación:
    ```bash
    sbatch -J vqe_S1001 -p qpu -t 00:30:00 -c 1 --mem=8G --export=ALL,SINGLE_SEED=1001 -o logs/vqe_qmio_seed_1001_%j.out -e logs/vqe_qmio_seed_1001_%j.err slurm/run_vqe_qmio.sh
    ```
    *(Los resultados se guardan de forma aislada en `resultados_qite_qmio_seed_{seed}.csv` y `resultados_vqe_qmio_seed_{seed}.csv`, y se fusionan posteriormente en local)*.

---

### Fase 4: Bucle de Validación Real en GPU (Cerrado)

Una vez que un resolvedor encuentra una solución óptima en el paisaje matemático QUBO:
1. **Deduplicación**: Se comprueba manualmente si la configuración sugerida por los resolvedores ya existe en el pool de ejecuciones preexistentes de `data/resultados_hpo.csv`. Si ya fue evaluada en la Fase 1, se reutiliza su resultado para ahorrar recursos computacionales.
2. **Entrenamiento y Validación en GPU**: Si la configuración óptima sugerida es un nuevo descubrimiento, se lanza un trabajo de entrenamiento en la cola GPU A100 del clúster (`short` partition) mediante `sbatch` ejecutando el wrapper `train_hpo.py`.
3. **Validación Rigurosa**: El Transformer se entrena bajo la configuración sugerida durante **exactamente 50 épocas a través de 5 folds (5-Fold CV)**.
4. **Aislamiento de Recursos**: Por restricciones del clúster CESGA, cada envío a las GPUs A100 solicita exactamente 1 GPU y 32 CPUs (`-c 32 --gres=gpu:a100:1`), asignando identificadores de checkpoint únicos para evitar colisiones.

*   **Ejemplo de comando de envío manual a Slurm:**
    ```bash
    sbatch -J tfg_train_opt -p short -t 04:00:00 -c 32 --mem=16G --gres=gpu:a100:1 \
      --wrap "module load python/3.10.8 gcc/12.3.0 && source ~/tfg_env/bin/activate && \
      python train_hpo.py --batch_size 64 --lr 5e-05 --emb_size 1024 --attn_heads 1 \
      --enc_layers 2 --dec_layers 4 --job_id opt_sol_64_1024_2_4"
    ```

---

### Fase 5: Sincronización de Métricas y Comparación Científica

Una vez finalizados los entrenamientos de validación de la Fase 4:
1. **Registro manual**: Se copian las métricas físicas reales obtenidas (Accuracy, Loss y Tiempo promedio de cómputo) del entrenamiento del Transformer al archivo maestro `data/resultados_hpo.csv` (y a los másteres de resultados de los resolvedores: `resultados_bayes_opt.csv`, `resultados_vqe_qmio.csv`, etc.), reemplazando los valores `-1.0` de las configuraciones recién entrenadas.
2. **Cruce de Datos**: Se genera la tabla máster consolidada en `resultados/resultados_soluciones_reales.csv` asociando a cada una de las 55 soluciones su energía QUBO, factibilidad y métricas del entrenamiento real.
3. **Visualización Interactiva**: Se despliega el dashboard Streamlit para analizar gráficamente y comparar el rendimiento de los solvers clásicos y cuánticos:
    ```bash
    streamlit run dashboard.py
    ```

---

## Fundamentos Físicos de las Optimización Cuánticas

Este repositorio incorpora soluciones avanzadas para suprimir el ruido stocástico e inestabilidades de Qiskit observadas en la ejecución física sobre hardware cuántico real:

*   **Unificación y Reducción del Ansatz a `reps=1`**: Reducimos el ansatz a `RealAmplitudes(reps=1)` con entrelazamiento lineal. Esto parametrizar el circuito con 60 ángulos y reduce las compuertas CNOT a exactamente **29 compuertas** (frente a las 58 CNOTs que requerían dos repeticiones). En Qmio (tasa de error CNOT ~1%), la fidelidad general se dispara de **55% a 75%**, evitando que el ruido estocástico difumine el paisaje QUBO y permitiendo que VQE minimice la energía reduciendo las violaciones One-Hot a solo 1 o 2 en hardware físico.
*   **Euler de Paso Fijo en VarQITE (`num_timesteps=10`)**: Por defecto, VarQITE dinámico de Qiskit calcula infinitesimalmente el paso temporal $dt$ adaptándose al ruido stocástico, lo que provoca la generación de más de 350,000 circuitos redundantes y Out Of Memory (OOM) en el frontend. Forzamos un resolvedor de paso fijo Euler con exactamente **10 pasos fijos de $dt=0.2$**, reduciendo la cola a solo 17,700 circuitos cuánticos. La emulación clásica se acelera un 300% y se erradica por completo la saturación de memoria.

---

## Resultados Consolidados en Hardware Real (QPU Qmio)

Los resultados máster unificados de la QPU física real de 32 cúbits bajo el mismo ansatz revelan la superioridad teórica de la evolución temporal imaginaria:

| Algoritmo de Optimización | Tasa de Factibilidad | Violaciones One-Hot Promedio | RMSE Promedio | Tiempo de Ejecución (Qpu) |
| :--- | :---: | :---: | :---: | :---: |
| **Optimización Bayesiana (Referencia)** | **100%** | **0.0** | **~0.005** | ~1.4 minutos |
| **VarQITE-VQE (QPU Real - 10 steps)** | **100%** | **0.0** | **~20.57** | **~3.7 minutos** |
| **VQE-COBYLA (QPU Real - reps=1)** | 0% | 2.4 | ~5.79 | ~26.0 minutos |

> [!TIP]
> **Conclusión Científica**: Gracias al principio de decaimiento no unitario en tiempo imaginario implementado en VarQITE, las penalizaciones de Lagrange actúan purificando de forma natural el estado de espín cuántico hacia el subespacio factible. Esto permite que **VarQITE alcance un 100% de tasa de factibilidad (cero violaciones One-Hot en todas las semillas evaluadas) directamente en la QPU superconductora real de Qmio**, superando drásticamente el comportamiento caótico y ruidoso de VQE-COBYLA estándar en hardware ruidoso (NISQ).

### Estado actual de la Fase de Validación Clásica-Cuántica
La fase experimental cuántica y clásica está **completada al 100% de forma definitiva** para todos los resolvedores. La comparativa consolidada en `resultados_soluciones_reales.csv` unifica un total de **55 soluciones óptimas**.

Actualmente, las últimas 3 combinaciones de hiperparámetros que faltan por validar físicamente en GPU (mostrando `-1.0` en los archivos maestros) se encuentran **programadas y encoladas con éxito en el programador de Slurm del CESGA** (Job IDs: `7300527`, `7300528`, `7300529`). Tras la conclusión del mantenimiento del clúster este martes 2 de junio, estos trabajos están en estado `PENDING` a la espera de que se liberen recursos de GPU A100. En cuanto finalicen de entrenarse, se registrarán sus métricas físicas reales en las tablas correspondientes completando el 100% de la comparativa.

