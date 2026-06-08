# Resultados e Imágenes (`resultados/`)

Esta carpeta compila las tablas de rendimiento experimental y almacena las representaciones gráficas en alta resolución exportadas por el módulo visualizador del proyecto.

## Archivos de Resultados (CSVs)

*   **`resultados_soluciones_reales.csv`**: Tabla unificada que consolida las 55 soluciones óptimas entrenadas de forma real en la GPU del CESGA para las 5 semillas, cruzando tiempo y precisión real.
*   **`resultados_bayes_opt.csv`**: Resultados experimentales de la optimización clásica basada en procesos gaussianos.
*   **`resultados_vqe_sweep.csv`**: Barrido experimental del impacto del ruido cuántico sobre el algoritmo VQE simulado en FT3.
*   **`resultados_qite_sweep.csv`**: Barrido experimental del impacto del ruido cuántico sobre el algoritmo VarQITE simulado en FT3.
*   **`resultados_vqe_qmio.csv`**: Fichero de telemetría cuántica obtenido físicamente en la QPU real de Qmio al ejecutar VQE.
*   **`resultados_qite_qmio.csv`**: Fichero de telemetría cuántica obtenido físicamente en la QPU real de Qmio al ejecutar VarQITE.

## Gráficos de Salida (PNGs)

*   **`frontera_pareto.png`**: Gráfico comparativo de dos paneles (Vista General y Zoom de la frontera) que evalúa la precisión media frente al tiempo medio de entrenamiento en GPU.
*   **`comparativa_tiempos.png`**: Gráfico comparativo de dos paneles que enfrenta el tiempo medio de búsqueda del solver cuántico/clásico frente al tiempo medio de entrenamiento real del Transformer.
*   **`comparativa_precision.png`**: Diagrama de barras independiente que compara la exactitud media real alcanzada por los Transformers afinados por cada algoritmo.
*   **`comparativa_energia.png`**: Muestra la energía normalizada media del QUBO lograda por cada resolvedor clásico/cuántico.
*   **`comparativa_energia_desglosada.png`**: Análisis de la paradoja de la factibilidad, comparando la energía de rendimiento raw (sin penalización) y la energía total del Hamiltoniano cuántico (con Lagrange $\lambda$).
*   **`comparativa_factibilidad.png`**: Tasa media de factibilidad cuántica en la QPU real.
*   **`impacto_ruido_factibilidad.png`**: Curva de degradación de la factibilidad conforme aumenta el ruido (Sin Ruido, Ruido Bajo, Medio y Alto) para VQE y VarQITE.
*   **`promedio_violaciones.png`**: Media del número de violaciones de la restricción cuántica One-Hot.
*   **`degradacion_tiempo_ruido.png`**: Tiempo medio consumido por los simuladores clásicos cuánticos en el FT3 al modelar el canal de ruido.
*   **`comparativa_tiempos_semillas.png`**: Tiempos de ejecución del optimizador detallados semilla a semilla.
