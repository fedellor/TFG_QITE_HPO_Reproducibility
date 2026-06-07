#!/usr/bin/env python3
# ==============================================================================
# SCRIPT: Code/generate_qubo.py
# DESCRIPCIÓN: Carga los resultados de entrenamiento clássicos, realiza el mapeo One-Hot
#              , ejecuta la regresión cuadrática regularizada (Lasso L1) y ensambla la 
#              matriz final Q (H, hamiltoniana) del modelo QUBO con penalizaciones de Lagrange.
# ==============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import json

# Espacio discreto de búsqueda formal de 30 cúbits (6 hiperparámetros x 5 opciones)
# Coincidente de forma exacta con el documento explicativo de los tutores
HPO_SPACE = {
    'BATCH_SIZE': [16, 32, 64, 128, 256],
    'LEARNING_RATE': [1e-3, 5e-4, 2e-4, 1e-4, 5e-5],
    'EMB_SIZE': [64, 128, 256, 512, 1024],
    'ATTN_HEADS': [1, 2, 4, 8, 16],
    'ENC_LAYERS': [1, 2, 3, 4, 6],
    'DEC_LAYERS': [1, 2, 3, 4, 6]
}

# Generar el mapeo de variables binarias (índices 0 a 29)
VAR_NAMES = []
VAR_MAP = {}
idx = 0
for hparam, options in HPO_SPACE.items():
    for opt in options:
        VAR_NAMES.append(f"{hparam}_{opt}")
        VAR_MAP[idx] = (hparam, opt)
        idx += 1

def find_closest_option(value, options):
    """
    Asigna un valor clásico de HPO a la opción discreta de referencia más cercana.
    """
    if isinstance(value, float):
        # Escala logarítmica para Learning Rate
        distances = [abs(np.log10(value) - np.log10(opt)) for opt in options]
    else:
        distances = [abs(value - opt) for opt in options]
    return options[np.argmin(distances)]

def encode_one_hot(row):
    """
    Convierte una ejecución HPO clásica en un vector binario de 30 cúbits.
    """
    x = np.zeros(30)
    for p_idx, (hparam, options) in enumerate(HPO_SPACE.items()):
        # Obtener el valor de la fila clásica correspondente
        val = row[hparam]
        closest = find_closest_option(val, options)
        opt_idx = options.index(closest)
        # Activar el bit correspondente dentro del grupo del hiperparámetro
        global_idx = p_idx * 5 + opt_idx
        x[global_idx] = 1.0
    return x

def build_quadratic_features(X_bin):
    """
    Construye la matriz de características lineales y de interacción cuadrática
    para la regresión de segundo orden: 30 lineales + 435 interacciones = 465 características.
    """
    N_samples = X_bin.shape[0]
    features = []
    feature_labels = []

    # 1. Agregar términos lineales (x_i)
    for i in range(30):
        features.append(X_bin[:, i])
        feature_labels.append(('linear', i))

    # 2. Agregar términos de acoplamiento cuadrático (x_i * x_j para i < j)
    for i in range(30):
        for j in range(i + 1, 30):
            features.append(X_bin[:, i] * X_bin[:, j])
            feature_labels.append(('quadratic', i, j))

    return np.column_stack(features), feature_labels

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == 'src':
        script_dir = os.path.dirname(script_dir)
    results_path = os.path.join(script_dir, 'data', 'resultados_hpo.csv')

    if not os.path.exists(results_path):
        print(f"ERROR: No se encuentra el archivo de resultados en '{results_path}'.")
        return

    # 1. Cargar el dataset de ejecuciones clásicas
    df = pd.read_csv(results_path)
    N_samples = len(df)

    # 2. Calcular la Puntuación de Rendimiento Multiobjetivo (Objetivo de la Regresión)
    # NOTA: Por requerimiento explícito de diseño del TFG, el LOSS se ignora completamente.
    # El objetivo es puramente multiobjetivo basado en MAXIMIZAR la precisión (Accuracy) 
    # y MINIMIZAR el tiempo de ejecución (Execution_Time_Seconds).
    beta = 0.3  # Peso del tiempo de ejecución (30% velocidad, 70% precisión)
    
    # Normalizar tiempo de ejecución
    time_min = df['Execution_Time_Seconds'].min()
    time_max = df['Execution_Time_Seconds'].max()
    time_norm = (df['Execution_Time_Seconds'] - time_min) / (time_max - time_min + 1e-9)
    
    # Función de coste real a minimizar (Ecuación Multiobjetivo del TFG)
    y = -(1 - beta) * df['Final_Accuracy'] + beta * time_norm
    
    # 3. Codificar las muestras experimentales a variables binarias
    X_bin_list = []
    for _, row in df.iterrows():
        X_bin_list.append(encode_one_hot(row))
    X_bin = np.vstack(X_bin_list)

    # 4. Construir las características cuadráticas de interacción
    X_quad, feat_labels = build_quadratic_features(X_bin)

    # 5. Ejecutar la regresión lineal regularizada Lasso (L1) con Validación Cruzada
    # Usamos cv=10, 1000 alphas en escala logarítmica, max_iter=100000, tol=1e-6 y selection='random'
    # para obtener el ajuste y generalización más óptimos posibles sobre el espacio discreto.
    lasso = LassoCV(
        cv=10,
        n_alphas=1000,
        max_iter=100000,
        tol=1e-6,
        selection='random',
        n_jobs=-1,
        random_state=42
    )
    lasso.fit(X_quad, y)

    # Calcular el RMSE del ajuste de la regresión cuadrática
    y_pred = lasso.predict(X_quad)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    # Extraer coeficientes
    coefs = lasso.coef_
    intercept = lasso.intercept_

    # Extraer h_i (lineales) y J_ij (cuadráticos)
    h = np.zeros(30)
    J = np.zeros((30, 30))

    active_couplings = 0
    for idx_f, label in enumerate(feat_labels):
        val = coefs[idx_f]
        if abs(val) > 1e-6:  # Coeficiente no nulo
            if label[0] == 'linear':
                i = label[1]
                h[i] = val
            elif label[0] == 'quadratic':
                i = label[1]
                j = label[2]
                J[i, j] = val
                J[j, i] = val # Simetría
                active_couplings += 1

    # 6. Ensamblar la Matriz QUBO Q con multiplicadores de Lagrange (Restricción One-Hot)
    # Seleccionamos un peso de penalización lambda robusto para obligar al sistema a cumplir las restricciones
    # Se calcula proporcionalmente a la escala de los coeficientes de la regresión
    max_coeff = max(np.max(np.abs(h)), np.max(np.abs(J))) if len(h) > 0 else 1.0
    lagrange_lambda = round(2.5 * max_coeff, 4)

    Q = np.zeros((30, 30))

    # Definir los rangos de las variables para cada uno de los 6 hiperparámetros
    # Grupo 0: 0-4, Grupo 1: 5-9, Grupo 2: 10-14, Grupo 3: 15-19, Grupo 4: 20-24, Grupo 5: 25-29
    # Para construir la matriz Hamiltoniana Q de forma matemáticamente exacta y simétrica:
    # - Diagonal: Q[i, i] = h[i] - lambda
    # - Fuera de la diagonal, mismo hiperparámetro (exclusión): Q[i, j] = 0.5 * J[i, j] + lambda
    # - Fuera de la diagonal, distinto hiperparámetro (interacción): Q[i, j] = 0.5 * J[i, j]
    # Esto asegura que al calcular x^T * Q * x, la energía coincida exactamente con la función 
    # cuadrática y penalizaciones sin duplicación de los coeficientes fuera de la diagonal.
    for i in range(30):
        # 1. Elementos de la diagonal (lineales)
        Q[i, i] = h[i] - lagrange_lambda
        
        for j in range(i + 1, 30):
            # Obtener el grupo (hiperparámetro) de cada variable
            group_i = i // 5
            group_j = j // 5
            
            if group_i == group_j:
                # 2. Elementos del mismo hiperparámetro (Exclusión estricta)
                Q[i, j] = 0.5 * J[i, j] + lagrange_lambda
                Q[j, i] = Q[i, j]
            else:
                # 3. Elementos de hiperparámetros distintos (Interacción real sin penalización)
                Q[i, j] = 0.5 * J[i, j]
                Q[j, i] = Q[i, j]

    # 7. Exportar los resultados en formatos estructurados
    output_q_csv = os.path.join(script_dir, 'data', 'matriz_qubo.csv')
    output_metrics_json = os.path.join(script_dir, 'data', 'metricas_qubo.json')

    # Guardar la matriz Q estrictamente en el CSV
    df_q = pd.DataFrame(Q, index=VAR_NAMES, columns=VAR_NAMES)
    df_q.to_csv(output_q_csv)

    # Guardar las métricas de rendimiento y metadatos estrictamente en el JSON
    meta_data = {
        'muestras_clasicas': int(N_samples),
        'mejor_alpha': float(lasso.alpha_),
        'rmse_regresion': float(rmse),
        'coeficientes_lineales_activos': int(np.count_nonzero(h)),
        'acoplamientos_cuadraticos_activos': int(active_couplings),
        'multiplicador_lagrange_lambda': float(lagrange_lambda),
        'hparam_space': HPO_SPACE,
        'variable_names': VAR_NAMES,
        'intercept': float(intercept),
        'beta_multiobjective': float(beta)
    }
    
    with open(output_metrics_json, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4)

    # Eliminar archivos obsoletos si existen para mantener limpio el directorio
    for old_file in ['matriz_qubo.json', 'metricas_qubo.csv']:
        old_path = os.path.join(script_dir, old_file)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except Exception:
                pass

    # Imprimir exclusivamente los resultados y el nombre del CSV generado
    print(f"Muestras clásicas cargadas: {N_samples} (Pérdida/Loss completamente ignorada)")
    print(f"Mejor coeficiente de regularización (alpha): {lasso.alpha_:.8f}")
    print(f"RMSE de la regresión Lasso L1: {rmse:.8f}")
    print(f"Coeficientes lineales activos: {np.count_nonzero(h)}")
    print(f"Acoplamientos cuadráticos activos: {active_couplings}")
    print(f"Multiplicador de Lagrange (lambda): {lagrange_lambda}")
    print(f"Nombre del CSV generado: matriz_qubo.csv")
    print(f"Métricas guardadas en: metricas_qubo.json")

if __name__ == '__main__':
    main()
