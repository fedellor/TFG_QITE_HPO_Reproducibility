#!/usr/bin/env python3
# ==============================================================================
# SCRIPT: Code/sampler_hpo.py
# DESCRIPCIÓN: Muestreo uniforme determinista de hiperparámetros con filtrado de integridad.
# ==============================================================================

import random
import csv
import json
import os

# Espacio discreto de búsqueda definido por el tutor
SPACE = {
    'BATCH_SIZE': [16, 32, 48, 64, 96, 128, 192, 256, 512, 1024],
    'LEARNING_RATE': [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 1e-6],
    'EMB_SIZE': [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
    'ATTN_HEADS': [1, 2, 4, 8, 16],
    'ENC_LAYERS': [1, 2, 3, 4, 5, 6, 8],
    'DEC_LAYERS': [1, 2, 3, 4, 5, 6, 8]
}

def generate_hpo_pool(seed=42, sample_size=300):
    """
    Genera el pool de hiperparámetros de forma determinista aplicando el filtro arquitectónico:
    EMB_SIZE % ATTN_HEADS == 0
    """
    # Establecer la semilla para máxima reproducibilidad en el muestreo
    random.seed(seed)
    
    valid_combinations = []
    
    # Generamos el producto cartesiano del espacio de búsqueda discreto
    for bs in SPACE['BATCH_SIZE']:
        for lr in SPACE['LEARNING_RATE']:
            for emb in SPACE['EMB_SIZE']:
                for heads in SPACE['ATTN_HEADS']:
                    for enc in SPACE['ENC_LAYERS']:
                        for dec in SPACE['DEC_LAYERS']:
                            # Filtro estricto de consistencia arquitectónica del Transformer
                            if emb % heads == 0:
                                valid_combinations.append({
                                    'BATCH_SIZE': bs,
                                    'LEARNING_RATE': lr,
                                    'EMB_SIZE': emb,
                                    'ATTN_HEADS': heads,
                                    'ENC_LAYERS': enc,
                                    'DEC_LAYERS': dec
                                })
    
    total_valid = len(valid_combinations)
    print(f"Total de combinaciones válidas en el hiperespacio: {total_valid} de {10*12*12*5*7*7}")
    
    if total_valid < sample_size:
        raise ValueError(f"El número de combinaciones válidas ({total_valid}) es menor al tamaño de muestra solicitado ({sample_size})")
    
    # Muestreo uniforme determinista usando seed
    sampled = random.sample(valid_combinations, sample_size)
    
    # Ordenamos para mayor consistencia visual
    sampled.sort(key=lambda x: (x['BATCH_SIZE'], x['LEARNING_RATE'], x['EMB_SIZE']))
    
    return sampled

def export_results(sampled):
    # Encontrar la ruta absoluta de la carpeta Code
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == 'src':
        script_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(script_dir, 'pool_jobs.csv')
    json_path = os.path.join(script_dir, 'pool_jobs.json')
    
    # Exportar a CSV indexado (1-based para SLURM Array Jobs)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Cabecera
        writer.writerow(['INDEX', 'BATCH_SIZE', 'LEARNING_RATE', 'EMB_SIZE', 'ATTN_HEADS', 'ENC_LAYERS', 'DEC_LAYERS'])
        for idx, cfg in enumerate(sampled, start=1):
            writer.writerow([
                idx,
                cfg['BATCH_SIZE'],
                cfg['LEARNING_RATE'],
                cfg['EMB_SIZE'],
                cfg['ATTN_HEADS'],
                cfg['ENC_LAYERS'],
                cfg['DEC_LAYERS']
            ])
            
    # Exportar a JSON de soporte estructurado indexado
    json_data = {}
    for idx, cfg in enumerate(sampled, start=1):
        json_data[str(idx)] = cfg
        
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)
        
    print(f"Pool de HPO exportado exitosamente a '{csv_path}' y '{json_path}'")

if __name__ == '__main__':
    sampled_pool = generate_hpo_pool(seed=42, sample_size=300)
    export_results(sampled_pool)
