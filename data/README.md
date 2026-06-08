# Gestión de Datos (`data/`)

Esta carpeta está dedicada a la persistencia de registros de eventos, matrices de acoplamiento cuadrático cuántico, metadatos y pesos neuronales preentrenados del EventTransformer.

## Archivos de Datos

*   **`resultados_hpo.csv`**: Historial maestro que recopila los resultados clásicos de HPO con 322 ejecuciones de validación cruzada y su rendimiento asociado en la GPU del clúster.
*   **`matriz_qubo.csv`**: Matriz binaria cuadrática simétrica $Q$ ($30 \times 30$) generada mediante regresión regularizada Lasso L1 LARS. Representa el Hamiltoniano de coste cuántico del espacio HPO.
*   **`metricas_qubo.json`**: Metadatos asociados al ajuste matemático del QUBO, almacenando el RMSE y el parámetro de penalización de Lagrange $\lambda$ utilizado para modelar la restricción cuántica One-Hot.

## Subdirectorios

*   **`models/`**:
    *   `env_permit_fold0_event_predictor.pth`: Pesos preentrenados en formato binario PyTorch correspondientes al EventTransformer para el pliegue 0 de validación.
*   **`event_log/`**:
    *   `attributes.yaml`: Definición de atributos de actividad, recursos y ciclo de vida de los logs.
    *   `env_permit.xes.gz`: Registro de eventos real en formato estructurado XES (comprimido en GZIP).
    *   `window_sizes.json`: Configuración de la ventana temporal de autoatención para secuencias del proceso.
    *   **`splitted/`**: Ficheros XES resultantes de la división en 5 folds para la validación cruzada (archivos comprimidos `train_foldX_env_permit.xes.gz`, `val_foldX_env_permit.xes.gz` y `test_foldX_env_permit.xes.gz`).
