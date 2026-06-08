# Módulo de Código Fuente (`src/`)

Esta carpeta contiene la implementación íntegra del pipeline algorítmico del proyecto, incluyendo el preprocesamiento de logs de eventos, la definición de la arquitectura Deep Learning, las rutinas de entrenamiento HPO clásico y el modelado y resolución cuántica del QUBO.

## Archivos del Módulo

*   **`config.py`**: Parámetros globales del sistema, rutas de persistencia en disco y semillas pseudoaleatorias reproducibles (`7, 42, 99, 123, 1001`).
*   **`embeddings.py`**: Capas de proyección continua en PyTorch para atributos categóricos y numéricos del log de eventos.
*   **`encoders_and_decoders.py`**: Bloques de autoatención multicabeza del `EventTransformer` (codificadores y decodificadores secuenciales).
*   **`output_layers.py`**: Capas lineales de salida para predicción multitarea (actividad siguiente, tiempo de ciclo restante e información de recursos).
*   **`training.py`**: Lógica de entrenamiento por lotes, optimizador AdamW, programación de tasa de aprendizaje y funciones de pérdida ponderadas.
*   **`evaluation.py`**: Cálculo de métricas de precisión mediante distancia de similitud de Damerau-Levenshtein sobre secuencias predictivas.
*   **`utils.py`**: Procesamiento de cadenas, vectorización, empaquetado de secuencias (*padding*) y funciones auxiliares de sistema.
*   **`sampler_hpo.py`**: Muestreador del espacio continuo-discreto HPO clásico bajo restricciones físicas y de paralelismo del clúster.
*   **`train_hpo.py`**: Envoltorio (*wrapper*) para el entrenamiento de pliegues cruzados (5-Fold CV, 50 épocas por fold) con persistencia segura mediante bloqueo de archivos (`fcntl`).
*   **`generate_qubo.py`**: Módulo de ajuste de regresión regularizada Lasso L1 LARS Cross-Validation que mapea el espacio de hiperparámetros a la matriz binaria cuadrática QUBO.
*   **`solve_bayes_opt.py`**: Resolvedor clásico mediante optimización bayesiana con Procesos Gaussianos.
*   **`solve_vqe.py`**: Resolvedor cuántico variacional VQE empleando simuladores MPS en CPU/GPU o conectividad física directa a la QPU superconductora de Qmio.
*   **`solve_qite.py`**: Resolvedor cuántico de evolución en tiempo imaginario VarQITE con integrador de Euler de paso fijo sobre la ecuación de McLachlan.
*   **`generate_plots.py`**: Script centralizado encargado de compilar los CSV experimentales y renderizar los 10 gráficos de alta resolución.
*   **`train.ipynb`**: Cuaderno Jupyter interactivo para pruebas rápidas y ejecuciones aisladas en GPU.

## Subdirectorios

*   **`data_processors/`**:
    *   `attributes.py`: Extractor de tipos de atributos de eventos categóricos e índices lógicos.
    *   `event_logs.py`: Mapeador lógico del registro de eventos estructurado en formato XES.
