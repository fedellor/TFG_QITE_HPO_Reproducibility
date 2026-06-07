#!/usr/bin/env python3
# ==============================================================================
# SCRIPT: Code/solve_bayes_opt.py
# DESCRIPCIÓN: Optimización Bayesiana clásica (gp_minimize) para resolver el QUBO
#              de 30 variables en 5 semillas secuenciales (1001, 123, 99, 42, 7).
# ==============================================================================

import os
import json
import time
import csv
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer

# --- Cargar la matriz QUBO y metadatos ---
def load_qubo():
    # Obtener el directorio absoluto del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == 'src':
        script_dir = os.path.dirname(script_dir)
    # Ruta del archivo que almacena la matriz QUBO
    csv_path = os.path.join(script_dir, 'data', 'matriz_qubo.csv')
    # Ruta del archivo JSON con la configuración del espacio de búsqueda
    json_path = os.path.join(script_dir, 'data', 'metricas_qubo.json')
    
    # Comprobar que los archivos necesarios existen
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        raise FileNotFoundError("Ejecute 'generate_qubo.py' primero.")
        
    # Leer el archivo JSON con codificación UTF-8
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # Cargar la matriz QUBO a partir del CSV
    Q = pd.read_csv(csv_path, index_col=0).values
    # Retornar los datos del QUBO, el espacio y lambda
    return Q, meta['hparam_space'], meta.get('multiplicador_lagrange_lambda', 0.0)

try:
    # Intentar cargar la matriz y parámetros
    Q, HPO_SPACE, LAGRANGE_LAMBDA = load_qubo()
    print(f"QUBO cargado | Lambda: {LAGRANGE_LAMBDA:.4f}")
except Exception as e:
    # Cargar valores por defecto en caso de fallo
    print(f"WARNING: {e}")
    Q = np.zeros((30, 30))
    HPO_SPACE = {}
    LAGRANGE_LAMBDA = 0.0

# --- Evaluar energía QUBO: E(x) = x^T * Q * x ---
# Esta función actúa como la caja negra u objetivo matemático a optimizar.
# Recibe un vector de espines binarios (30 variables de decisión binarias {0, 1})
# y calcula el valor de la energía hamiltoniana asociada mediante la forma cuadrática
# E(x) = x^T * Q * x, donde Q es la matriz del problema modelada con penalizaciones L1/L2.
def evaluate_qubo(x):
    # Calcular la multiplicación de la matriz por el vector binario
    return float(np.array(x, dtype=float).T @ Q @ np.array(x, dtype=float))

# --- Decodificar vector binario de 30 bits a hiperparámetros HPO ---
# Mapea el vector binario (compuesto por 6 sub-bloques One-Hot de 5 bits cada uno)
# de vuelta a los valores físicos correspondientes en el espacio de hiperparámetros.
# Para cada hiperparámetro, se aplica argmax sobre su respectivo bloque de 5 variables
# binarias para obtener el índice de la opción seleccionada.
def decode_vector(x_bin):
    # Nombres de los hiperparámetros del Transformer
    hparam_keys = ['BATCH_SIZE', 'LEARNING_RATE', 'EMB_SIZE', 'ATTN_HEADS', 'ENC_LAYERS', 'DEC_LAYERS']
    # Reconstruir el diccionario mapeado usando argmax
    return {hp: HPO_SPACE.get(hp, [0,1,2,3,4])[np.argmax(x_bin[i*5 : (i+1)*5])] for i, hp in enumerate(hparam_keys)}

# --- Ejecutar gp_minimize (Optimización Bayesiana) ---
# Emplea Procesos Gaussianos para modelar de forma probabilística la superficie de costo de QUBO.
# Utiliza una función de adquisición (por defecto 'GP-UCB' o 'EI') para balancear la exploración de
# regiones con alta incertidumbre y la explotación de mínimos locales encontrados.
# Parámetros clave:
#   - dimensions: Espacio discreto entero binario de 30 variables [0, 1].
#   - n_calls=80: Número máximo de evaluaciones en la función objetivo.
#   - n_initial_points=20: 20 muestras aleatorias iniciales como a priori para ajustar el Proceso Gaussiano.
#   - random_state=seed: Garantiza la reproducibilidad y el rigor estadístico sobre las 5 semillas.
def run_bayes_opt(seed):
    # Registrar el tiempo de inicio de la optimización
    start_time = time.time()
    # Ejecutar gp_minimize con el optimizador bayesiano
    res = gp_minimize(
        func=evaluate_qubo,
        dimensions=[Integer(0, 1) for _ in range(30)],
        n_calls=80,
        n_initial_points=20,
        random_state=seed,
        n_jobs=1
    )
    # Retornar el valor óptimo, el vector y el tiempo transcurrido
    return float(res.fun), res.x, time.time() - start_time

def main():
    # Semillas aleatorias utilizadas en el estudio
    seeds = [1001, 123, 99, 42, 7]
    # Comprobar si se solicita ejecutar una única semilla
    single_seed = os.environ.get('SINGLE_SEED')
    if single_seed:
        seeds = [int(single_seed)]
    results = []
    
    print("\n>>> EJECUTANDO OPTIMIZACIÓN BAYESIANA CLÁSICA SOBRE QUBO <<<")
    # Iterar secuencialmente sobre todas las semillas
    for seed in seeds:
        try:
            # Ejecutar el optimizador para la semilla actual
            energy, x_bin, dt = run_bayes_opt(seed)
            # Decodificar el vector binario resultante
            cfg = decode_vector(x_bin)
            # Contar las violaciones a la restricción One-Hot
            violations = sum(1 for g in range(6) if sum(x_bin[g * 5 : (g + 1) * 5]) != 1)
            # Calcular la energía del QUBO corregida sumando la penalización de Lagrange
            energy_corrected = energy + 6.0 * LAGRANGE_LAMBDA
            
            # Registrar los resultados obtenidos en la lista
            results.append({
                'seed': seed, 'energy': energy, 'energy_corrected': energy_corrected,
                'config': cfg, 'violations': violations, 'is_feasible': 1 if violations == 0 else 0, 'time': dt
            })
            print(f"Semilla {seed:4d} finalizada: Energía QUBO = {energy:.4f} | Violaciones: {violations}")
        except Exception as e:
            print(f"Semilla {seed:4d} falló: {e}")

    # Guardar en resultados/resultados_bayes_opt.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == 'src':
        script_dir = os.path.dirname(script_dir)
    output_csv = os.path.join(script_dir, 'resultados', 'resultados_bayes_opt.csv')
    # Crear el directorio de resultados si no existe
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Escribir los resultados en formato CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Cabecera del archivo CSV de salida
        writer.writerow([
            'Semilla', 'Energia_QUBO', 'Energia_Real_Estimada', 'Tiempo_Segundos', 'Violaciones_OneHot',
            'Factible', 'Explorada', 'Precision_Real_HPO', 'Tiempo_Real_HPO',
            'BATCH_SIZE', 'LEARNING_RATE', 'EMB_SIZE', 'ATTN_HEADS', 'ENC_LAYERS', 'DEC_LAYERS', 'RMSE'
        ])
        for r in results:
            cfg = r['config']
            # Escribir fila de datos correspondiente
            writer.writerow([
                r['seed'], round(r['energy'], 4), round(r['energy_corrected'], 4), round(r['time'], 3),
                r['violations'], r['is_feasible'], 0, -1.0, -1.0,  # Valores HPO por defecto
                cfg['BATCH_SIZE'], cfg['LEARNING_RATE'], cfg['EMB_SIZE'], cfg['ATTN_HEADS'],
                cfg['ENC_LAYERS'], cfg['DEC_LAYERS'], -1.0
            ])
    print(f"Resultados exportados a: {output_csv}\n")

if __name__ == '__main__':
    main()
