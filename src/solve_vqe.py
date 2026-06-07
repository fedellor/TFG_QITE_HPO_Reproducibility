#!/usr/bin/env python3
# ==============================================================================
# SCRIPT: Code/solve_vqe.py
# DESCRIPCIÓN: Resuelve el QUBO (30 qubits) usando el algoritmo variacional VQE
#              (SamplingVQE) en Qiskit. Ejecuciones simuladas (con/sin ruido) o en QPU.
# ==============================================================================

import os
import sys
import time
import csv
import json
import numpy as np

# Evitar colisión de importación con config.py local al usar qmiotools/qmio
sys.path = [p for p in sys.path if p not in ('', '.', os.getcwd())]
os.environ['ZMQ_SERVER'] = os.environ.get('ZMQ_SERVER', 'dummy')

try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import Sampler
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("[WARNING] Qiskit no instalado. Se usará simulación clásica de fallback.")

# --- Cargar matriz QUBO y metadatos ---
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
    import pandas as pd
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

# --- Obtener sampler cuántico ---
def get_sampler(noise_mode, g1q=0.0, g2q=0.0, ro=0.0):
    # Comprobar si se solicita usar la QPU real de Qmio
    if noise_mode == 'qmio':
        from qmiotools.integrations.qiskitqmio import QmioBackend, FakeQmio
        from qiskit.primitives import BackendSampler
        # Instanciar el backend físico o simulado de Qmio
        backend = QmioBackend() if os.environ.get('USE_REAL_QPU') == '1' else FakeQmio()
        # Retornar el Sampler para Qmio
        return BackendSampler(backend=backend)
    
    # Importar el simulador de Aer por defecto
    from qiskit_aer.primitives import Sampler as AerSampler
    # Configurar el sampler clásico de Aer usando MPS
    sam = AerSampler(backend_options={"method": "matrix_product_state", "matrix_product_state_max_bond_dimension": 128})
    # Aplicar el modelo de ruido en escenarios no ideales
    if noise_mode != 'ideal':
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
        noise = NoiseModel()
        # Agregar errores en puertas de un cúbit
        if g1q > 0: noise.add_all_qubit_quantum_error(depolarizing_error(g1q, 1), ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
        # Agregar errores en puertas de dos cúbits (CX)
        if g2q > 0: noise.add_all_qubit_quantum_error(depolarizing_error(g2q, 2), ['cx'])
        # Agregar errores en la lectura de cúbits
        if ro > 0: noise.add_all_readout_error(ReadoutError([[1.0 - ro, ro], [ro, 1.0 - ro]]))
        # Establecer opciones y shots del simulador
        sam.set_options(noise_model=noise, shots=8192)
    return sam

# --- Construir QuadraticProgram representando la matriz Q ---
def build_qp():
    # Inicializar el QuadraticProgram de Qiskit
    qp = QuadraticProgram("TFG_HPO_QUBO")
    # Agregar las 30 variables de decisión binarias
    vars = [qp.binary_var(f"var_{i}") for i in range(30)]
    # Añadir los términos lineales a la función objetivo
    qp.minimize(linear={f"var_{i}": float(Q[i, i]) for i in range(30)})
    # Añadir los términos cuadráticos a la función objetivo
    qp.minimize(quadratic={(f"var_{i}", f"var_{j}"): 2.0 * float(Q[i, j]) for i in range(30) for j in range(i+1, 30) if abs(Q[i, j]) > 1e-6})
    return qp

# --- Decodificar vector binario a hiperparámetros HPO ---
def decode_vector(x_bin):
    # Nombres de los hiperparámetros del Transformer
    hparam_keys = ['BATCH_SIZE', 'LEARNING_RATE', 'EMB_SIZE', 'ATTN_HEADS', 'ENC_LAYERS', 'DEC_LAYERS']
    # Reconstruir el diccionario mapeado usando argmax
    return {hp: HPO_SPACE.get(hp, [0,1,2,3,4])[np.argmax(x_bin[i*5 : (i+1)*5])] for i, hp in enumerate(hparam_keys)}

# --- Ejecución del optimizador VQE ---
# VQE (Variational Quantum Eigensolver) busca aproximar el estado fundamental (mínima energía)
# del Hamiltoniano Ising que mapea el QUBO.
# El algoritmo combina:
#   1. Un circuito parametrizado (Ansatz): RealAmplitudes con reps=1 y entrelazamiento lineal.
#   2. Un optimizador clásico (COBYLA): Optimiza los ángulos del ansatz basándose en la energía medida.
#   3. Un Sampler cuántico (Backend/Simulación): Mide el valor esperado de la energía en la QPU o simulador.
def run_vqe(seed, noise_mode='ideal', g1q=0.0, g2q=0.0, ro=0.0):
    # Comprobar si Qiskit está instalado o usar fallback clásico
    if not HAS_QISKIT:
        np.random.seed(seed)
        start_time = time.time()
        x = np.zeros(30)
        for g in range(6): x[g * 5 + np.random.randint(0, 5)] = 1.0
        return float(x.T @ Q @ x), x, time.time() - start_time

    # Construir el modelo cuadrático (QuadraticProgram)
    qp = build_qp()
    
    # Ansatz: El circuito parametrizado para generar el estado cuántico de prueba |psi(theta)>
    # reps=1 y entanglement='linear' minimizan la profundidad y el número de puertas CNOT
    # para reducir el efecto del ruido cuántico depolarizante en la QPU superconductora.
    ansatz = RealAmplitudes(num_qubits=30, reps=1, entanglement='linear')
    
    # Optimizador clásico COBYLA (Constrained Optimization BY Linear Approximations)
    # Es libre de gradientes, lo que lo hace idóneo para mitigar las fluctuaciones del ruido de la QPU.
    optimizer = COBYLA(maxiter=100)
    
    # Punto inicial aleatorio uniforme para los parámetros del circuito theta
    np.random.seed(seed)
    initial_point = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    
    # Resolver usando SamplingVQE (adecuado para observables diagonales como QUBO/Ising)
    sampler = get_sampler(noise_mode, g1q, g2q, ro)
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point)
    
    # MinimumEigenOptimizer mapea el QuadraticProgram al problema cuántico de Ising
    # y traduce el resultado binario obtenido de vuelta al espacio clásico.
    optimizer_vqe = MinimumEigenOptimizer(vqe)
    
    start_time = time.time()
    result = optimizer_vqe.solve(qp)
    return float(result.fval), np.array(result.x), time.time() - start_time

def main():
    seeds = [1001, 123, 99, 42, 7]
    single_seed = os.environ.get('SINGLE_SEED')
    if single_seed:
        seeds = [int(single_seed)]
        
    noise_mode = os.environ.get('NOISE_MODE', 'sweep').lower()
    if noise_mode == 'sweep':
        scenarios = [
            ("ideal", 0.0, 0.0, 0.0, "Sin ruido (Ideal)"),
            ("low", 0.0001, 0.001, 0.002, "Ruido bajo (1q=0.01%, 2q=0.1%, ro=0.2%)"),
            ("medium", 0.0005, 0.005, 0.005, "Ruido medio (1q=0.05%, 2q=0.5%, ro=0.5%)"),
            ("high", 0.001, 0.010, 0.010, "Ruido alto (1q=0.1%, 2q=1.0%, ro=1.0%)")
        ]
    else:
        scenarios = [(noise_mode, 0.0005, 0.005, 0.005, f"Escenario: {noise_mode}")]

    depth, total_gates, cnots = 0, 0, 0
    if HAS_QISKIT:
        try:
            ansatz_dec = RealAmplitudes(num_qubits=30, reps=1, entanglement='linear').decompose()
            depth = ansatz_dec.depth()
            ops = ansatz_dec.count_ops()
            total_gates = sum(ops.values())
            cnots = ops.get('cx', 0)
        except Exception as e:
            print(f"Error calculando complejidad del ansatz: {e}")

    all_results = []
    for mode, g1q, g2q, ro, info in scenarios:
        print(f"\n>>> INICIANDO ESCENARIO VQE: {info.upper()} (Semillas: {seeds}) <<<")
        for seed in seeds:
            try:
                energy, x_bin, dt = run_vqe(seed, mode, g1q, g2q, ro)
                cfg = decode_vector(x_bin)
                violations = sum(1 for g in range(6) if sum(x_bin[g * 5 : (g + 1) * 5]) != 1)
                energy_corrected = energy + 6.0 * LAGRANGE_LAMBDA
                
                all_results.append({
                    'seed': seed, 'energy': energy, 'energy_corrected': energy_corrected,
                    'config': cfg, 'violations': violations, 'is_feasible': 1 if violations == 0 else 0,
                    'time': dt, 'mode': mode, 'info': info
                })
                print(f"   Semilla {seed:4d} finalizada: Energía QUBO = {energy:.4f} | Violaciones: {violations}")
            except Exception as e:
                print(f"   Semilla {seed:4d} falló: {e}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == 'src':
        script_dir = os.path.dirname(script_dir)
    single_seed = os.environ.get('SINGLE_SEED')
    if single_seed:
        output_csv = os.path.join(script_dir, 'resultados', f'resultados_vqe_{noise_mode}_seed_{single_seed}.csv')
    else:
        output_csv = os.path.join(script_dir, 'resultados', f'resultados_vqe_{noise_mode}.csv')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Semilla', 'Energia_QUBO', 'Energia_Real_Estimada', 'Tiempo_Segundos', 'Violaciones_OneHot',
            'Factible', 'Explorada', 'Precision_Real_HPO', 'Tiempo_Real_HPO',
            'BATCH_SIZE', 'LEARNING_RATE', 'EMB_SIZE', 'ATTN_HEADS', 'ENC_LAYERS', 'DEC_LAYERS', 'RMSE',
            'Profundidad_Circuito', 'Total_Puertas', 'Puertas_CNOT', 'Modelo_Ruido', 'Escenario_Modo'
        ])
        for r in all_results:
            cfg = r['config']
            writer.writerow([
                r['seed'], round(r['energy'], 4), round(r['energy_corrected'], 4), round(r['time'], 3),
                r['violations'], r['is_feasible'], 0, -1.0, -1.0,  # Valores HPO por defecto
                cfg['BATCH_SIZE'], cfg['LEARNING_RATE'], cfg['EMB_SIZE'], cfg['ATTN_HEADS'],
                cfg['ENC_LAYERS'], cfg['DEC_LAYERS'], -1.0,
                depth, total_gates, cnots, r['info'], r['mode']
            ])
    print(f"\nResultados consolidados guardados en: {output_csv}")

if __name__ == '__main__':
    main()
