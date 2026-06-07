#!/usr/bin/env python3
# ==============================================================================
# SCRIPT: Code/train_hpo.py
# DESCRIPCIÓN: Wrapper HPO con validación cruzada de 5 folds en GPU (CUDA) y fcntl.
# ==============================================================================

import argparse
import os
import sys
import csv
import torch
import time

# Obtener rutas absolutas y registrar la carpeta Code/ en el PATH de Python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
CODE_DIR = SCRIPT_DIR
if os.path.basename(CODE_DIR) == 'src':
    CODE_DIR = os.path.dirname(CODE_DIR)

# Importamos config y corregimos la ruta de datos a absoluta antes de importar los demás módulos
import config
config.ROOT_DATA_PATH = os.path.join(CODE_DIR, 'data') + '/'

from data_processors.event_logs import EventLog, get_window_size
from encoders_and_decoders import EventTransformer
from training import fit, calculate_loss, generate_dataloader
from evaluation import test

# Importamos fcntl para bloqueo exclusivo de archivos en entornos UNIX (FT3)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    print("[WARNING] fcntl no está disponible. Se usará escritura estándar (sin bloqueo).")

def append_results(csv_path, row):
    """
    Escribe una fila en el archivo CSV usando bloqueo exclusivo (fcntl).
    """
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        if HAS_FCNTL:
            # Bloqueo exclusivo del descriptor de archivo
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            if not file_exists:
                # Escribimos cabeceras si el archivo es nuevo
                writer.writerow([
                    'BATCH_SIZE', 'LEARNING_RATE', 'EMB_SIZE', 'ATTN_HEADS',
                    'ENC_LAYERS', 'DEC_LAYERS', 'Final_Accuracy', 'Final_Loss', 'Execution_Time_Seconds'
                ])
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())  # Forzar sincronización física del almacenamiento
        finally:
            if HAS_FCNTL:
                # Liberar bloqueo
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def main():
    # Obtener por defecto el CSV dentro de la carpeta Code
    default_csv = os.path.join(CODE_DIR, 'data', 'resultados_hpo.csv')

    parser = argparse.ArgumentParser(description="Wrapper HPO con 5-Fold Cross-Validation en GPU")
    parser.add_argument('--batch_size', type=int, required=True, help="Tamaño de batch")
    parser.add_argument('--lr', type=float, required=True, help="Tasa de aprendizaje")
    parser.add_argument('--emb_size', type=int, required=True, help="Tamaño de embeddings")
    parser.add_argument('--attn_heads', type=int, required=True, help="Número de cabezas de atención")
    parser.add_argument('--enc_layers', type=int, required=True, help="Capas del encoder")
    parser.add_argument('--dec_layers', type=int, required=True, help="Capas del decoder")
    parser.add_argument('--epochs', type=int, default=50, help="Épocas por fold (default: 50)")
    parser.add_argument('--log_name', type=str, default='env_permit', help="Nombre del registro de eventos")
    parser.add_argument('--results_csv', type=str, default=default_csv, help="Ruta del CSV de salida")
    parser.add_argument('--job_id', type=str, default='standalone', help="ID de la tarea paralela")
    args = parser.parse_args()

    # Comprobación de reanudación: verificar si esta configuración ya ha sido evaluada y registrada
    if os.path.exists(args.results_csv):
        try:
            with open(args.results_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is not None:
                    # Buscar si la combinación de hiperparámetros ya está en el archivo resultados
                    for row in reader:
                        if len(row) >= 6:
                            try:
                                bs = int(row[0])
                                lr = float(row[1])
                                emb = int(row[2])
                                heads = int(row[3])
                                enc = int(row[4])
                                dec = int(row[5])
                                if (bs == args.batch_size and 
                                    abs(lr - args.lr) < 1e-9 and 
                                    emb == args.emb_size and 
                                    heads == args.attn_heads and 
                                    enc == args.enc_layers and 
                                    dec == args.dec_layers):
                                    print(f"\n[RESUME] Config BS={bs}, LR={lr}, EMB={emb}, HEADS={heads}, ENC={enc}, DEC={dec} ALREADY COMPLETED in results CSV. Skipping execution!")
                                    sys.exit(0)
                            except ValueError:
                                continue
        except Exception as e:
            print(f"[WARNING] Error reading results CSV for resume check: {e}")

    print(f"\n=== Iniciando Experimento HPO #{args.job_id} (5-Fold CV sobre GPU) ===")
    print(f"Hiperparámetros a Evaluar: BS={args.batch_size}, LR={args.lr}, EMB={args.emb_size}, "
          f"HEADS={args.attn_heads}, ENC={args.enc_layers}, DEC={args.dec_layers}, Epochs/Fold={args.epochs}")

    # Configuración del dispositivo de cómputo (priorizar aceleración CUDA en GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de cómputo asignado: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    start_time = time.time()
    fold_accuracies = []
    fold_losses = []
    folds = ['0', '1', '2', '3', '4']

    for fold in folds:
        print(f"\n--- [Experimento #{args.job_id}] Entrenando Fold {fold}/4 ---")
        
        # Ponemos la semilla aleatoria para garantizar reproducibilidad en las divisiones y pesos iniciales
        config.set_seed(42)

        # Cargar el registro de eventos (EventLog) correspondiente al fold actual
        event_log = EventLog(args.log_name, fold, start_of_suffix=True)
        window_size = get_window_size('auto', event_log)
        log_dict = event_log.to_dict()

        # Instanciar el EventTransformer con los parámetros de la arquitectura seleccionada
        model = EventTransformer(
            cat_attributes=log_dict['cat_attributes'],
            num_attributes=log_dict['num_attributes'],
            embedding_size=args.emb_size,
            encoder_layers=args.enc_layers,
            decoder_layers=args.dec_layers,
            encoder_attn_heads=args.attn_heads,
        )
        model.to(device)

        # Crear un directorio exclusivo para almacenar checkpoints y prevenir colisiones entre hilos paralelos
        unique_store_path = os.path.join(config.ROOT_DATA_PATH, 'models_hpo', f"job_{args.job_id}_fold_{fold}")
        os.makedirs(unique_store_path, exist_ok=True)

        # Ajuste y entrenamiento del modelo para el número de épocas especificado
        model = fit(
            model,
            event_log,
            window_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            stopping_patience=args.epochs + 5, # Deshabilitar Early Stopping temprano para forzar el número total de épocas solicitado
            store_path=unique_store_path,
            force_new_model=True,
            lr=args.lr,
            clip_loss=True,
            stop_with_loss=True
        )

        # Evaluar el modelo en el conjunto de validación al completar el entrenamiento
        model.eval()
        val_dataloader = generate_dataloader(event_log, device, window_size, batch_size=args.batch_size, validation=True)
        
        with torch.no_grad():
            val_loss = calculate_loss(model, event_log, val_dataloader, clip_loss=False, validation=True)

        # Calcular la similitud de Damerau-Levenshtein para las secuencias de actividades predichas
        similarities = test(model, event_log, window_size, validation=True, batch_size=args.batch_size * 4)
        val_accuracy = similarities.get('concept:name', 0.0)

        print(f"Fold {fold} finalizado: Accuracy = {val_accuracy:.6f}, Loss = {val_loss:.6f}")
        
        fold_accuracies.append(val_accuracy)
        fold_losses.append(val_loss)

        # Liberar la memoria de vídeo (GPU VRAM) explícitamente para evitar fragmentación y errores OOM
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    end_time = time.time()
    total_time = end_time - start_time

    # Calcular los promedios globales de exactitud y pérdida a través de los 5 folds analizados
    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    mean_loss = sum(fold_losses) / len(fold_losses)
    
    print(f"\n>>> Experimento #{args.job_id} Completado (Promedio 5 Folds) <<<")
    print(f"Acc Promedio: {mean_accuracy:.6f} | Loss Promedio: {mean_loss:.6f} | Tiempo Total: {total_time:.2f} s")

    # Persistencia atómica de las métricas obtenidas utilizando fcntl para evitar condiciones de carrera en disco
    row = [
        args.batch_size,
        args.lr,
        args.emb_size,
        args.attn_heads,
        args.enc_layers,
        args.dec_layers,
        round(mean_accuracy, 6),
        round(mean_loss, 6),
        round(total_time, 2)
    ]
    
    print(f"Escribiendo resultados de forma atómica en: {args.results_csv}")
    append_results(args.results_csv, row)
    print(f"=== Experimento HPO #{args.job_id} Finalizado con Éxito ===\n")

if __name__ == '__main__':
    main()
