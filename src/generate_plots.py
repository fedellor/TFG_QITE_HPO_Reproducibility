import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def main():
    local_root = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(local_root) == 'src':
        local_root = os.path.dirname(local_root)
    
    # Load HPO metadata and QUBO matrix for Raw Energy calculation
    with open(os.path.join(local_root, 'data', 'metricas_qubo.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    hparam_space = meta['hparam_space']
    Q = pd.read_csv(os.path.join(local_root, 'data', 'matriz_qubo.csv'), index_col=0).values
    
    def find_closest_option(value, options):
        if isinstance(value, float):
            distances = [abs(np.log10(value) - np.log10(opt)) for opt in options]
        else:
            distances = [abs(value - opt) for opt in options]
        return options[np.argmin(distances)]
        
    def get_raw_energy(row):
        x = np.zeros(30)
        for p_idx, hparam in enumerate(['BATCH_SIZE', 'LEARNING_RATE', 'EMB_SIZE', 'ATTN_HEADS', 'ENC_LAYERS', 'DEC_LAYERS']):
            val = row[hparam]
            options = hparam_space[hparam]
            closest = find_closest_option(val, options)
            opt_idx = options.index(closest)
            global_idx = p_idx * 5 + opt_idx
            x[global_idx] = 1.0
        return float(x.T @ Q @ x)
        
    res_dir = os.path.join(local_root, "resultados")
    os.makedirs(res_dir, exist_ok=True)

    # 1. Load the CSVs
    bayes_csv = os.path.join(res_dir, "resultados_bayes_opt.csv")
    vqe_qmio_csv = os.path.join(res_dir, "resultados_vqe_qmio.csv")
    qite_qmio_csv = os.path.join(res_dir, "resultados_qite_qmio.csv")
    vqe_sweep_csv = os.path.join(res_dir, "resultados_vqe_sweep.csv")
    qite_sweep_csv = os.path.join(res_dir, "resultados_qite_sweep.csv")

    # Read dataframes
    bayes_df = pd.read_csv(bayes_csv)
    vqe_qmio_df = pd.read_csv(vqe_qmio_csv)
    qite_qmio_df = pd.read_csv(qite_qmio_csv)
    vqe_sweep_df = pd.read_csv(vqe_sweep_csv)
    qite_sweep_df = pd.read_csv(qite_sweep_csv)

    # --- PLOT 1: EXTREME TIME OPPOSITION (SOLVER VS HPO MEAN) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # Solver execution times (Subplot 1)
    ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
    vqe_ideal_t = vqe_sweep_df[vqe_sweep_df['Escenario_Modo'] == 'ideal']['Tiempo_Segundos'].mean()
    vqe_qpu_t = vqe_qmio_df['Tiempo_Segundos'].mean()
    qite_ideal_t = qite_sweep_df[qite_sweep_df['Escenario_Modo'] == 'ideal']['Tiempo_Segundos'].mean()
    qite_qpu_t = qite_qmio_df['Tiempo_Segundos'].mean()
    bayes_t = bayes_df['Tiempo_Segundos'].mean()

    categories = [
        'Opt. Bayesiana\n(Clásica)', 
        'VQE\n(Sim. Ideal FT3)', 
        'VQE\n(QPU Qmio Real)', 
        'VarQITE\n(Sim. Ideal FT3)', 
        'VarQITE\n(QPU Qmio Real)'
    ]
    times_solver = [bayes_t, vqe_ideal_t, vqe_qpu_t, qite_ideal_t, qite_qpu_t]
    colors = ['#2ca02c', '#aec7e8', '#1f77b4', '#ffbb78', '#ff7f0e']

    bars1 = ax1.bar(categories, times_solver, color=colors, edgecolor='black', width=0.55, zorder=3)
    ax1.set_yscale('log')
    ax1.set_ylabel('Tiempo de Ejecución del Solver (Segundos - Escala Log)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Tiempo de Optimización / Búsqueda (Solver)', fontsize=12, fontweight='bold', pad=12)
    ax1.tick_params(axis='x', labelsize=10)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, height * 1.15, f'{height:.1f} s', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # HPO training times (Subplot 2)
    ax2.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    bayes_hpo_t = bayes_df['Tiempo_Real_HPO'].mean()
    vqe_ideal_hpo_t = vqe_sweep_df[vqe_sweep_df['Escenario_Modo'] == 'ideal']['Tiempo_Real_HPO'].mean()
    vqe_qpu_hpo_t = vqe_qmio_df['Tiempo_Real_HPO'].mean()
    qite_ideal_hpo_t = qite_sweep_df[qite_sweep_df['Escenario_Modo'] == 'ideal']['Tiempo_Real_HPO'].mean()
    qite_qpu_hpo_t = qite_qmio_df['Tiempo_Real_HPO'].mean()
    
    times_hpo = [bayes_hpo_t, vqe_ideal_hpo_t, vqe_qpu_hpo_t, qite_ideal_hpo_t, qite_qpu_hpo_t]
    
    bars2 = ax2.bar(categories, times_hpo, color=colors, edgecolor='black', width=0.55, zorder=3)
    ax2.set_ylabel('Tiempo Real de Entrenamiento de Transformers (Segundos)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Tiempo Medio de Entrenamiento Real en GPU (5 Folds)', fontsize=12, fontweight='bold', pad=12)
    ax2.tick_params(axis='x', labelsize=10)
    
    max_hpo_t = max(times_hpo)
    ax2.set_ylim(0, max_hpo_t * 1.15)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, height + (max_hpo_t * 0.02), f'{height:.1f} s', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    fig.suptitle('Análisis Comparativo de Tiempos: Optimización de Hiperparámetros vs. Entrenamiento Real', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "comparativa_tiempos.png"), dpi=300)
    plt.close()

    # --- PLOT 2: ENERGY COMPARISON (QUBO QUALITY) ---
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    bayes_e = bayes_df['Energia_QUBO'].mean()
    vqe_qpu_e = vqe_qmio_df['Energia_QUBO'].mean()
    # Correct QITE energy by subtracting the Ising offset to put it on the same scale
    qite_qpu_e = qite_qmio_df['Energia_QUBO'].mean() - 19.9473
    
    categories_e = ['Opt. Bayesiana (Clásica)', 'VQE (QPU Real Qmio)', 'VarQITE (QPU Real Qmio)']
    energies = [bayes_e, vqe_qpu_e, qite_qpu_e]
    colors_e = ['#2ca02c', '#d62728', '#ff7f0e']

    bars = plt.bar(categories_e, energies, color=colors_e, edgecolor='black', width=0.5, zorder=3)
    plt.ylabel('Energía del QUBO (Menor es mejor)', fontsize=12, fontweight='bold')
    plt.title('Energía Media del QUBO Alcanzada por cada Algoritmo (Normalizada)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(fontsize=11, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height - 0.3 if height < 0 else height + 0.1, f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "comparativa_energia.png"), dpi=300)
    plt.close()

    # --- PLOT 3: FEASIBILITY RATE (ONE-HOT VIOLATIONS) ---
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    bayes_f = bayes_df['Factible'].mean() * 100
    vqe_qpu_f = vqe_qmio_df['Factible'].mean() * 100
    qite_qpu_f = qite_qmio_df['Factible'].mean() * 100

    categories_f = ['Opt. Bayesiana', 'VQE (QPU Real)', 'VarQITE (QPU Real)']
    rates = [bayes_f, vqe_qpu_f, qite_qpu_f]
    colors_f = ['#2ca02c', '#d62728', '#ff7f0e']

    bars = plt.bar(categories_f, rates, color=colors_f, edgecolor='black', width=0.5, zorder=3)
    plt.ylabel('Tasa de Factibilidad (% de soluciones sin violar One-Hot)', fontsize=12, fontweight='bold')
    plt.title('Tasa de Factibilidad (QUBO One-Hot) en la QPU Física', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 110)
    plt.xticks(fontsize=11, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "comparativa_factibilidad.png"), dpi=300)
    plt.close()

    # --- PLOT 4: NOISE IMPACT ON FEASIBILITY SWEEP ---
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

    # Calculate average feasibility rate per noise level for VQE and QITE
    noise_scenarios = ['ideal', 'low', 'medium', 'high']
    noise_labels = ['Ideal (Sin Ruido)', 'Ruido Bajo', 'Ruido Medio', 'Ruido Alto']
    
    vqe_feasibility = []
    qite_feasibility = []
    
    for sc in noise_scenarios:
        v_f = vqe_sweep_df[vqe_sweep_df['Escenario_Modo'] == sc]['Factible'].mean() * 100
        q_f = qite_sweep_df[qite_sweep_df['Escenario_Modo'] == sc]['Factible'].mean() * 100
        vqe_feasibility.append(v_f)
        qite_feasibility.append(q_f)

    plt.plot(noise_labels, vqe_feasibility, marker='o', linewidth=2.5, color='#d62728', label='VQE (COBYLA)', zorder=3)
    plt.plot(noise_labels, qite_feasibility, marker='s', linewidth=2.5, color='#ff7f0e', label='VarQITE (McLachlan)', zorder=3)
    
    plt.ylabel('Tasa de Factibilidad (% de soluciones factibles)', fontsize=12, fontweight='bold')
    plt.xlabel('Nivel de Ruido en Simulación', fontsize=12, fontweight='bold')
    plt.title('Impacto del Ruido en la Factibilidad: VQE vs. VarQITE', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(-5, 105)
    plt.legend(fontsize=11)
    plt.xticks(fontsize=11, fontweight='bold')
    
    for i, txt in enumerate(vqe_feasibility):
        plt.annotate(f'{txt:.0f}%', (noise_labels[i], vqe_feasibility[i] - 6), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='#d62728')
    for i, txt in enumerate(qite_feasibility):
        plt.annotate(f'{txt:.0f}%', (noise_labels[i], qite_feasibility[i] + 2), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='#ff7f0e')

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "impacto_ruido_factibilidad.png"), dpi=300)
    plt.close()

    # --- PLOT 5: AVERAGE ONE-HOT VIOLATIONS ---
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

    # Average violations across runs (Simulated vs Real QPU)
    vqe_ideal_v = vqe_sweep_df[vqe_sweep_df['Escenario_Modo'] == 'ideal']['Violaciones_OneHot'].mean()
    vqe_qpu_v = vqe_qmio_df['Violaciones_OneHot'].mean()
    
    qite_ideal_v = qite_sweep_df[qite_sweep_df['Escenario_Modo'] == 'ideal']['Violaciones_OneHot'].mean()
    qite_qpu_v = qite_qmio_df['Violaciones_OneHot'].mean()

    categories_v = ['VQE (Sim. Ideal)', 'VQE (QPU Real)', 'VarQITE (Sim. Ideal)', 'VarQITE (QPU Real)']
    violations = [vqe_ideal_v, vqe_qpu_v, qite_ideal_v, qite_qpu_v]
    colors_v = ['#aec7e8', '#1f77b4', '#ffbb78', '#ff7f0e']

    bars = plt.bar(categories_v, violations, color=colors_v, edgecolor='black', width=0.6, zorder=3)
    plt.ylabel('Promedio de Violaciones One-Hot (Menor es mejor)', fontsize=12, fontweight='bold')
    plt.title('Promedio de Violaciones One-Hot: Simulación vs. QPU Real', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(fontsize=11, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.05, f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "promedio_violaciones.png"), dpi=300)
    plt.close()

    # --- PLOT 6: EVOLUTION OF SIMULATION AND QPU TIMES PER SEED ---
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # Sort data by seed to align properly
    bayes_sorted = bayes_df.sort_values(by='Semilla')
    vqe_qmio_sorted = vqe_qmio_df.sort_values(by='Semilla')
    qite_qmio_sorted = qite_qmio_df.sort_values(by='Semilla')
    
    vqe_sweep_ideal = vqe_sweep_df[vqe_sweep_df['Escenario_Modo'] == 'ideal'].sort_values(by='Semilla')
    qite_sweep_ideal = qite_sweep_df[qite_sweep_df['Escenario_Modo'] == 'ideal'].sort_values(by='Semilla')

    seeds_x = [f"Semilla {s}" for s in bayes_sorted['Semilla'].values]

    plt.plot(seeds_x, bayes_sorted['Tiempo_Segundos'], marker='o', linewidth=2, color='#2ca02c', label='Opt. Bayesiana (Clásica)', zorder=3)
    plt.plot(seeds_x, vqe_sweep_ideal['Tiempo_Segundos'], marker='v', linestyle='--', linewidth=2, color='#aec7e8', label='VQE (Simulación Ideal FT3)', zorder=3)
    plt.plot(seeds_x, vqe_qmio_sorted['Tiempo_Segundos'], marker='^', linewidth=2, color='#1f77b4', label='VQE (QPU Real Qmio)', zorder=3)
    plt.plot(seeds_x, qite_sweep_ideal['Tiempo_Segundos'], marker='<', linestyle='--', linewidth=2, color='#ffbb78', label='VarQITE (Simulación Ideal FT3)', zorder=3)
    plt.plot(seeds_x, qite_qmio_sorted['Tiempo_Segundos'], marker='>', linewidth=2, color='#ff7f0e', label='VarQITE (QPU Real Qmio)', zorder=3)

    plt.yscale('log')
    plt.ylabel('Tiempo de Ejecución (Segundos - Escala Log)', fontsize=12, fontweight='bold')
    plt.xlabel('Semillas del Experimento', fontsize=12, fontweight='bold')
    plt.title('Comparativa de Tiempos de Ejecución por Semilla', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10, loc='best')
    plt.xticks(fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "comparativa_tiempos_semillas.png"), dpi=300)
    plt.close()

    # --- PLOT 7: NOISE IMPACT ON SIMULATION TIME SWEEP ---
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

    noise_scenarios = ['ideal', 'low', 'medium', 'high']
    noise_labels = ['Ideal (Sin Ruido)', 'Ruido Bajo', 'Ruido Medio', 'Ruido Alto']

    vqe_means = []
    vqe_mins = []
    vqe_maxes = []
    qite_means = []
    qite_mins = []
    qite_maxes = []

    for sc in noise_scenarios:
        vqe_sc = vqe_sweep_df[vqe_sweep_df['Escenario_Modo'] == sc]
        qite_sc = qite_sweep_df[qite_sweep_df['Escenario_Modo'] == sc]
        
        vqe_means.append(vqe_sc['Tiempo_Segundos'].mean())
        vqe_mins.append(vqe_sc['Tiempo_Segundos'].min())
        vqe_maxes.append(vqe_sc['Tiempo_Segundos'].max())
        
        qite_means.append(qite_sc['Tiempo_Segundos'].mean())
        qite_mins.append(qite_sc['Tiempo_Segundos'].min())
        qite_maxes.append(qite_sc['Tiempo_Segundos'].max())

    plt.plot(noise_labels, vqe_means, marker='o', linewidth=2.5, color='#d62728', label='VQE (Simulación FT3)', zorder=3)
    plt.fill_between(noise_labels, vqe_mins, vqe_maxes, color='#d62728', alpha=0.15, zorder=2)
    
    plt.plot(noise_labels, qite_means, marker='s', linewidth=2.5, color='#ff7f0e', label='VarQITE (Simulación FT3)', zorder=3)
    plt.fill_between(noise_labels, qite_mins, qite_maxes, color='#ff7f0e', alpha=0.15, zorder=2)

    plt.yscale('log')
    plt.ylabel('Tiempo de Ejecución (Segundos - Escala Log)', fontsize=12, fontweight='bold')
    plt.xlabel('Nivel de Ruido en Simulación', fontsize=12, fontweight='bold')
    plt.title('Impacto del Ruido en el Tiempo de Simulación (FT3)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=11)
    plt.xticks(fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "degradacion_tiempo_ruido.png"), dpi=300)
    plt.close()

    # --- PLOT 8: DETAILED ENERGY COMPARISON (RAW VS TOTAL WITH PENALTIES) ---
    plt.figure(figsize=(11, 6.5))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # Compute raw and total energies
    bayes_total = bayes_df['Energia_QUBO'].mean()
    bayes_raw = bayes_df.apply(get_raw_energy, axis=1).mean()

    vqe_total = vqe_qmio_df['Energia_QUBO'].mean()
    vqe_raw = vqe_qmio_df.apply(get_raw_energy, axis=1).mean()

    qite_total = qite_qmio_df['Energia_QUBO'].mean() - 19.9473
    qite_raw = qite_qmio_df.apply(get_raw_energy, axis=1).mean()

    categories_d = ['Opt. Bayesiana (Clásica)', 'VQE (QPU Real Qmio)', 'VarQITE (QPU Real Qmio)']
    raw_energies = [bayes_raw, vqe_raw, qite_raw]
    total_energies = [bayes_total, vqe_total, qite_total]

    x = np.arange(len(categories_d))
    width = 0.35

    rects1 = plt.bar(x - width/2, raw_energies, width, label='Energía de Rendimiento Raw (Sin Penalización)', color='#1f77b4', edgecolor='black', zorder=3)
    rects2 = plt.bar(x + width/2, total_energies, width, label='Energía Total (con Penalización de Lagrange)', color='#d62728', edgecolor='black', zorder=3)

    plt.ylabel('Energía del QUBO (Menor es mejor)', fontsize=12, fontweight='bold')
    plt.title('Paradoja de la Factibilidad: Rendimiento Raw vs. Energía Total con Penalización', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, categories_d, fontsize=11, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    
    # Add values on top/bottom of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            va_align = 'bottom' if height >= 0 else 'top'
            xytext_y = 3 if height >= 0 else -14
            plt.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, xytext_y),
                        textcoords="offset points",
                        ha='center', va=va_align,
                        fontsize=9.5, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "comparativa_energia_desglosada.png"), dpi=300)
    plt.close()

    # --- PLOT 9: PARETO FRONTIER (ACCURACY VS TIME) ---
    plt.figure(figsize=(10.5, 6.5))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

    reales_csv = os.path.join(res_dir, "resultados_soluciones_reales.csv")
    reales_df = pd.read_csv(reales_csv)

    # Define groups for custom scatter coloring and shape
    groups = {
        'bayesiana': {
            'df': reales_df[reales_df['Algoritmo'].str.contains('Bayesiana', case=False)],
            'color': '#2ca02c', 'marker': 'o', 'label': 'Opt. Bayesiana (Clásica)'
        },
        'vqe-qmio': {
            'df': reales_df[reales_df['Algoritmo'].str.contains('VQE físico|VQE fisico', case=False)],
            'color': '#1f77b4', 'marker': '^', 'label': 'VQE QPU (Qmio Real)'
        },
        'vqe-ft3': {
            'df': reales_df[reales_df['Algoritmo'].str.contains('Simulación VQE Sweep|Simulacion VQE Sweep', case=False)],
            'color': '#aec7e8', 'marker': 'v', 'label': 'VQE FT3 (Simulación)'
        },
        'qite-qmio': {
            'df': reales_df[reales_df['Algoritmo'].str.contains('VarQITE físico|VarQITE fisico', case=False)],
            'color': '#ff7f0e', 'marker': '>', 'label': 'VarQITE QPU (Qmio Real)'
        },
        'qite-ft3': {
            'df': reales_df[reales_df['Algoritmo'].str.contains('Simulación VarQITE Sweep|Simulacion VarQITE Sweep', case=False)],
            'color': '#ffbb78', 'marker': '<', 'label': 'VarQITE FT3 (Simulación)'
        }
    }

    # Plot scatter points
    for name, g in groups.items():
        df_g = g['df']
        if not df_g.empty:
            plt.scatter(
                df_g['Execution_Time_Seconds'], df_g['Final_Accuracy'],
                color=g['color'], marker=g['marker'], s=85, edgecolor='black', alpha=0.85,
                label=g['label'], zorder=3
            )

    # Compute Pareto Frontier
    # Minimize time (x), Maximize accuracy (y)
    all_points = list(zip(reales_df['Execution_Time_Seconds'].values, reales_df['Final_Accuracy'].values))
    # Sort first by time ascending, then by accuracy descending
    sorted_points = sorted(all_points, key=lambda x: (x[0], -x[1]))
    pareto_front = []
    max_acc = -1
    for pt in sorted_points:
        if pt[1] > max_acc:
            pareto_front.append(pt)
            max_acc = pt[1]

    # Draw Pareto Line and efficient solutions highlight
    if pareto_front:
        pareto_x, pareto_y = zip(*pareto_front)
        plt.plot(
            pareto_x, pareto_y, '--', color='#d62728', linewidth=2.5, alpha=0.9,
            label='Frontera de Pareto (Eficiente)', zorder=4
        )
        
        plt.scatter(
            pareto_x, pareto_y, facecolors='none', edgecolors='#d62728', s=180,
            linewidths=2.0, zorder=5, label='Solución Pareto-Optimal'
        )

    plt.xlabel('Tiempo de Entrenamiento Real del Transformer (Segundos)', fontsize=12, fontweight='bold')
    plt.ylabel('Precisión Real del Transformer (Accuracy)', fontsize=12, fontweight='bold')
    plt.title('Frontera de Pareto: Compromiso entre Precisión Real y Tiempo de Entrenamiento', fontsize=13, fontweight='bold', pad=15)
    plt.legend(fontsize=9.5, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "frontera_pareto.png"), dpi=300)
    plt.close()

    # --- PLOT 10: ACCURACY COMPARISON (HPO QUALITY SUMMARY) ---
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

    categories_acc = [
        'Opt. Bayesiana\n(Clásica)', 
        'VQE\n(Sim. Ideal FT3)', 
        'VQE\n(QPU Qmio Real)', 
        'VarQITE\n(Sim. Ideal FT3)', 
        'VarQITE\n(QPU Qmio Real)'
    ]
    accuracies = [
        bayes_df['Precision_Real_HPO'].mean(),
        vqe_sweep_df[vqe_sweep_df['Escenario_Modo'] == 'ideal']['Precision_Real_HPO'].mean(),
        vqe_qmio_df['Precision_Real_HPO'].mean(),
        qite_sweep_df[qite_sweep_df['Escenario_Modo'] == 'ideal']['Precision_Real_HPO'].mean(),
        qite_qmio_df['Precision_Real_HPO'].mean()
    ]
    colors_acc = ['#2ca02c', '#aec7e8', '#1f77b4', '#ffbb78', '#ff7f0e']

    bars_acc = plt.bar(categories_acc, accuracies, color=colors_acc, edgecolor='black', width=0.55, zorder=3)
    plt.ylabel('Precisión Media Real del Transformer (Accuracy)', fontsize=12, fontweight='bold')
    plt.title('Precisión Media Real de los Transformers Ajustados por cada Algoritmo', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 1.05)
    plt.xticks(fontsize=10, fontweight='bold')
    
    for bar in bars_acc:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height*100:.2f}%', ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "comparativa_precision.png"), dpi=300)
    plt.close()

    print("All 10 plots generated successfully in: resultados/")

if __name__ == "__main__":
    main()
