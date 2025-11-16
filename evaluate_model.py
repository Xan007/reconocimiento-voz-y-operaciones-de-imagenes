"""
Evaluador de Modelos - Calcula el margen de error
Ingresa todos los audios de cada comando como si fueran entradas,
verifica si la predicción coincide con el comando real (por nombre de carpeta),
y calcula el porcentaje de acierto.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from processing.audio_utils import read_wav, pad_or_trim, normalize_audio
from processing.fft_utils import analyze_signal
from recognition.recognizer import load_models, compare_with_models
from config import COMANDOS_DIR, MODELOS_DIR

def evaluate_model():
    """
    Evalúa el modelo usando todos los archivos de entrenamiento.
    
    Algoritmo:
    1. Para cada comando (carpeta en data/comandos/):
        a) Cargar todas las grabaciones
        b) Para cada grabación:
            - Calcular energía (como si fuera entrada)
            - Comparar con modelos
            - Registrar si la predicción fue correcta
    2. Calcular porcentaje de acierto total
    3. Mostrar resultados por comando y general
    """
    
    print("\n" + "="*70)
    print("EVALUADOR DE MODELOS - CÁLCULO DE MARGEN DE ERROR")
    print("="*70)
    
    # Verificar que existen modelos
    if not os.path.isdir(MODELOS_DIR):
        print("❌ No existe la carpeta de modelos. Entrena primero con trainer.py")
        return
    
    model_files = [f for f in os.listdir(MODELOS_DIR) if f.endswith('.json')]
    if not model_files:
        print("❌ No hay modelos entrenados. Ejecuta trainer.py primero.")
        return
    
    print(f"\n✓ Modelos encontrados: {len(model_files)}")
    for mf in sorted(model_files):
        print(f"   - {mf}")
    
    # Cargar modelos
    try:
        models = load_models()
        print(f"\n✓ {len(models)} modelo(s) cargado(s)")
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return
    
    # Estructura para almacenar resultados
    results = {
        'por_comando': {},  # {comando: {'total': N, 'aciertos': N, 'errores': N}}
        'predicciones': []  # [{'comando_real': X, 'prediccion': Y, 'correcto': bool, 'archivo': F}]
    }
    
    # Procesar cada comando
    command_dirs = sorted([d for d in os.listdir(COMANDOS_DIR)
                          if os.path.isdir(os.path.join(COMANDOS_DIR, d))])
    
    if not command_dirs:
        print("❌ No hay carpetas de comandos. Graba datos primero.")
        return
    
    print(f"\n✓ Comandos encontrados: {len(command_dirs)}")
    for cmd in command_dirs:
        print(f"   - {cmd}")
    
    print("\n" + "="*70)
    print("EVALUANDO GRABACIONES")
    print("="*70)
    
    total_aciertos = 0
    total_evaluaciones = 0
    
    for comando_real in command_dirs:
        comando_path = os.path.join(COMANDOS_DIR, comando_real)
        wav_files = sorted([f for f in os.listdir(comando_path) if f.lower().endswith('.wav')])
        
        if not wav_files:
            print(f"\n⚠️  {comando_real.upper()}: Sin grabaciones")
            results['por_comando'][comando_real] = {
                'total': 0,
                'aciertos': 0,
                'errores': 0,
                'porcentaje': 0.0
            }
            continue
        
        aciertos = 0
        print(f"\n📂 Evaluando comando: {comando_real.upper()}")
        print(f"   Grabaciones: {len(wav_files)}")
        print("   " + "-"*66)
        
        for i, fname in enumerate(wav_files, 1):
            full_path = os.path.join(comando_path, fname)
            
            try:
                # Leer y preparar audio (igual que en reconocimiento)
                x, _ = read_wav(full_path)
                x = normalize_audio(x)
                x = pad_or_trim(x)
                
                # Calcular energía
                _, _, energies = analyze_signal(x)
                
                # Comparar con modelos
                prediccion, diffs = compare_with_models(energies, models)
                
                # Verificar si fue correcto
                correcto = (prediccion == comando_real)
                if correcto:
                    aciertos += 1
                    total_aciertos += 1
                
                total_evaluaciones += 1
                
                # Registrar predicción
                estado = "✓" if correcto else "✗"
                distancia_real = diffs.get(comando_real, float('inf'))
                
                print(f"   [{i:2d}/{len(wav_files)}] {estado} {fname:25s} → {prediccion.upper() if prediccion else 'ERROR':6s} "
                      f"(real: {distancia_real:.4f})", end="")
                
                if prediccion and prediccion != comando_real:
                    dist_predicha = diffs.get(prediccion, float('inf'))
                    print(f" pred: {dist_predicha:.4f}")
                else:
                    print()
                
                results['predicciones'].append({
                    'comando_real': comando_real,
                    'prediccion': prediccion,
                    'correcto': correcto,
                    'archivo': fname,
                    'distancia_real': float(distancia_real),
                    'distancias_todas': {k: float(v) for k, v in diffs.items()}
                })
                
            except Exception as e:
                print(f"   [{i:2d}/{len(wav_files)}] ✗ ERROR procesando {fname}: {e}")
                results['predicciones'].append({
                    'comando_real': comando_real,
                    'prediccion': None,
                    'correcto': False,
                    'archivo': fname,
                    'error': str(e)
                })
                total_evaluaciones += 1
        
        # Resumen por comando
        porcentaje = (aciertos / len(wav_files) * 100) if wav_files else 0
        errores = len(wav_files) - aciertos
        
        print("   " + "-"*66)
        print(f"   Resumen: {aciertos}/{len(wav_files)} aciertos ({porcentaje:.1f}%)")
        
        results['por_comando'][comando_real] = {
            'total': len(wav_files),
            'aciertos': aciertos,
            'errores': errores,
            'porcentaje': porcentaje
        }
    
    # Resumen general
    print("\n" + "="*70)
    print("RESUMEN GENERAL")
    print("="*70)
    
    for comando, stats in results['por_comando'].items():
        if stats['total'] > 0:
            print(f"{comando.upper():10s} {stats['aciertos']:2d}/{stats['total']:2d} "
                  f"({stats['porcentaje']:5.1f}%)")
    
    if total_evaluaciones > 0:
        porcentaje_general = (total_aciertos / total_evaluaciones * 100)
        print("-"*70)
        print(f"{'TOTAL':10s} {total_aciertos:2d}/{total_evaluaciones:2d} "
              f"({porcentaje_general:.1f}%)")
    else:
        porcentaje_general = 0.0
        print("❌ No se evaluaron grabaciones")
    
    print("="*70)
    
    # Preparar resultados
    results['total_aciertos'] = total_aciertos
    results['total_evaluaciones'] = total_evaluaciones
    results['porcentaje_acierto'] = porcentaje_general
    
    return results


def show_confusion_matrix(results):
    """
    Muestra una matriz de confusión de las predicciones.
    """
    print("\n" + "="*70)
    print("MATRIZ DE CONFUSIÓN")
    print("="*70)
    
    # Obtener lista de comandos únicos
    comandos = sorted(set(p['comando_real'] for p in results['predicciones']))
    predicciones_unicas = sorted(set(p['prediccion'] for p in results['predicciones'] if p['prediccion']))
    todos_los_comandos = sorted(set(comandos + predicciones_unicas))
    
    # Crear matriz
    matriz = {}
    for real in todos_los_comandos:
        matriz[real] = {}
        for pred in todos_los_comandos:
            matriz[real][pred] = 0
    
    # Llenar matriz
    for pred in results['predicciones']:
        real = pred['comando_real']
        prediccion = pred['prediccion'] or 'ERROR'
        if real in matriz and prediccion in todos_los_comandos:
            matriz[real][prediccion] += 1
    
    # Mostrar matriz
    ancho_col = 8
    print(f"\n{'Real \\ Pred':10s}", end="")
    for cmd in todos_los_comandos:
        print(f"{cmd.upper()[:7]:>8s}", end="")
    print()
    
    print("-" * (10 + len(todos_los_comandos) * 8))
    
    for real in todos_los_comandos:
        print(f"{real.upper()[:9]:10s}", end="")
        for pred in todos_los_comandos:
            print(f"{matriz[real][pred]:8d}", end="")
        print()
    
    print("="*70)


if __name__ == "__main__":
    try:
        results = evaluate_model()
        if results:
            show_confusion_matrix(results)
            print("\n✅ Evaluación completada")
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
