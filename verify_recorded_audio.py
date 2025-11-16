"""
Verifica que los audios grabados tienen las propiedades correctas:
- Duracion exacta de 1 segundo
- Frecuencia de muestreo de 44100 Hz
- Numero de muestras correcto (44100)
- Analiza las bandas de energia
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path
import soundfile as sf
from config import FS, DURATION, TARGET_N
from processing.fft_utils import analyze_signal

def verify_audio_file(filepath):
    """Verifica propiedades de un archivo de audio."""
    try:
        data, fs = sf.read(filepath)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        
        n_samples = len(data)
        duration = n_samples / fs
        
        # Calcula energias por banda
        _, _, energies = analyze_signal(data)
        
        return {
            'file': Path(filepath).name,
            'fs': fs,
            'n_samples': n_samples,
            'duration': duration,
            'expected_samples': int(FS * DURATION),
            'energies': energies,
            'main_band_energy': energies[0],
            'total_energy': np.sum(energies),
            'valid': fs == FS and n_samples == int(FS * DURATION)
        }
    except Exception as e:
        return {
            'file': Path(filepath).name,
            'error': str(e),
            'valid': False
        }

def main():
    """Verifica todos los audios del comando 'uno'."""
    print("\n" + "="*80)
    print("VERIFICACION DE AUDIOS GRABADOS - COMANDO 'UNO'")
    print("="*80)
    
    uno_path = Path("data/comandos/uno")
    if not uno_path.exists():
        print(f"ERROR: No existe {uno_path}")
        return
    
    audio_files = sorted(uno_path.glob("*.wav"))
    if not audio_files:
        print("No hay archivos WAV en data/comandos/uno/")
        return
    
    print(f"\nEncontrados {len(audio_files)} archivos de audio\n")
    
    results = []
    all_valid = True
    
    for i, filepath in enumerate(audio_files, 1):
        result = verify_audio_file(str(filepath))
        results.append(result)
        
        if 'error' in result:
            print(f"[{i:2d}] {result['file']:20s} - ERROR: {result['error']}")
            all_valid = False
        else:
            valid_marker = "[OK]" if result['valid'] else "[FAIL]"
            print(f"{valid_marker} [{i:2d}] {result['file']:20s} | "
                  f"FS={result['fs']:5d} Hz | "
                  f"Muestras={result['n_samples']:5d} ({result['duration']:.3f}s) | "
                  f"Banda1={result['main_band_energy']:.4f} | "
                  f"Total E={result['total_energy']:.4f}")
            
            if not result['valid']:
                all_valid = False
                if result['fs'] != FS:
                    print(f"        ADVERTENCIA: FS incorrecta (esperado {FS}, obtenido {result['fs']})")
                if result['n_samples'] != result['expected_samples']:
                    print(f"        ADVERTENCIA: Muestras incorrectas (esperado {result['expected_samples']}, obtenido {result['n_samples']})")
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    
    valid_count = sum(1 for r in results if r.get('valid', False))
    total_count = len(results)
    
    print(f"\nArchivos validos: {valid_count}/{total_count}")
    print(f"Duracion esperada: {DURATION} segundo(s) = {TARGET_N} muestras @ {FS} Hz")
    
    if all_valid and valid_count == total_count:
        print("\n[OK] Todos los audios son validos!")
        
        # Estadisticas de las bandas
        print("\nESTADISTICAS DE BANDAS:")
        print("-" * 80)
        
        all_energies = np.array([r['energies'] for r in results if r.get('valid')])
        
        for band_idx in range(all_energies.shape[1]):
            band_energies = all_energies[:, band_idx]
            print(f"Banda {band_idx}: Media={np.mean(band_energies):.6f}, "
                  f"Min={np.min(band_energies):.6f}, "
                  f"Max={np.max(band_energies):.6f}, "
                  f"Std={np.std(band_energies):.6f}")
        
        print("\nDistribucion principal de energia:")
        print("-" * 80)
        for band_idx in range(min(3, all_energies.shape[1])):
            band_energies = all_energies[:, band_idx]
            mean_energy = np.mean(band_energies)
            print(f"Banda {band_idx}: {mean_energy*100:.2f}%")
    else:
        print("\n[FAIL] Algunos audios no son validos. Revisa los errores arriba.")

if __name__ == "__main__":
    main()
