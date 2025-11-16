"""
Herramienta interactiva para grabar datos de entrenamiento.
Registra comandos de voz a 44100 Hz con duración de 1 segundo cada uno.
"""

import os
import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from config import FS, DURATION, CHANNELS

def get_available_devices():
    """Obtiene lista de dispositivos de audio disponibles."""
    devices = sd.query_devices()
    return devices

def select_device():
    """Permite al usuario seleccionar un dispositivo de audio."""
    devices = get_available_devices()
    print("\n" + "="*60)
    print("DISPOSITIVOS DE AUDIO DISPONIBLES:")
    print("="*60)
    
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append(i)
            print(f"[{len(input_devices)-1}] {device['name']} "
                  f"(canales: {device['max_input_channels']}, "
                  f"FS nativa: {int(device['default_samplerate'])} Hz)")
    
    if not input_devices:
        print("❌ No hay dispositivos de entrada disponibles")
        return None
    
    while True:
        try:
            choice = int(input(f"\nSelecciona dispositivo (0-{len(input_devices)-1}): "))
            if 0 <= choice < len(input_devices):
                device_id = input_devices[choice]
                print(f"\n✓ Dispositivo seleccionado: {devices[device_id]['name']}")
                return device_id
        except ValueError:
            pass
        print("❌ Opción inválida")

def get_next_filename(command_name):
    """Obtiene el siguiente número de archivo disponible."""
    base_path = Path(f"data/comandos/{command_name}")
    base_path.mkdir(parents=True, exist_ok=True)
    
    files = list(base_path.glob("grabacion_*.wav"))
    if not files:
        return "grabacion_001.wav", 1
    
    # Extrae números de nombres existentes
    numbers = []
    for f in files:
        try:
            num = int(f.stem.split('_')[1])
            numbers.append(num)
        except:
            pass
    
    next_num = max(numbers) + 1 if numbers else 1
    return f"grabacion_{next_num:03d}.wav", next_num

def record_audio(device_id, duration=DURATION, fs=FS, channels=CHANNELS):
    """Graba audio del dispositivo especificado."""
    print(f"\n🎤 Grabando {duration} segundo(s) a {fs} Hz...")
    print("   Habla ahora...")
    
    try:
        # Graba audio
        audio = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=channels,
            device=device_id,
            dtype=np.float32
        )
        sd.wait()  # Espera a que termine la grabación
        
        print("✓ Grabación completada")
        return audio.squeeze() if channels == 1 else audio
    except Exception as e:
        print(f"❌ Error durante la grabación: {e}")
        return None

def normalize_audio(audio, target_rms=0.1):
    """Normaliza el audio a un nivel RMS específico."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio * (target_rms / rms)
    # Previene saturación
    audio = np.clip(audio, -0.95, 0.95)
    return audio

def save_audio(audio, command_name, filename):
    """Guarda el audio en archivo WAV."""
    filepath = Path(f"data/comandos/{command_name}/{filename}")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        sf.write(str(filepath), audio, FS, subtype='PCM_16')
        print(f"✓ Guardado: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error al guardar: {e}")
        return False

def preview_audio(audio, fs=FS):
    """Reproduce el audio grabado."""
    print("\n🔊 Reproduciendo...")
    sd.play(audio, samplerate=fs)
    sd.wait()
    print("✓ Reproducción completada")

def main():
    """Función principal - loop de grabación interactivo."""
    print("\n" + "="*60)
    print("GRABADOR DE DATOS DE ENTRENAMIENTO")
    print("="*60)
    print(f"Configuración:")
    print(f"  • Frecuencia de muestreo: {FS} Hz")
    print(f"  • Duración por grabación: {DURATION} segundo(s)")
    print(f"  • Canales: {CHANNELS} (mono)")
    print(f"  • Ubicación: data/comandos/")
    
    # Selecciona dispositivo
    device_id = select_device()
    if device_id is None:
        print("\n❌ No se pudo seleccionar dispositivo")
        return
    
    print("\n" + "="*60)
    print("COMANDOS DISPONIBLES PARA GRABAR:")
    print("="*60)
    print("[1] UNO   - Graba ejemplos del comando 'uno'")
    print("[2] DOS   - Graba ejemplos del comando 'dos'")
    print("[3] TRES  - Graba ejemplos del comando 'tres'")
    print("[4] SALIR - Termina la grabación")
    
    while True:
        command_choice = input("\n¿Qué comando quieres grabar? (1-4): ").strip()
        
        if command_choice == "4":
            print("\n✓ Grabación finalizada")
            break
        elif command_choice not in ["1", "2", "3"]:
            print("❌ Opción inválida")
            continue
        
        command_name = "uno" if command_choice == "1" else ("dos" if command_choice == "2" else "tres")
        
        print(f"\n{'='*60}")
        print(f"GRABANDO COMANDO: '{command_name.upper()}'")
        print(f"{'='*60}")
        
        while True:
            filename, num = get_next_filename(command_name)
            print(f"\nGrabación #{num} para '{command_name}'")
            print("Opciones:")
            print("  [G] Grabar")
            print("  [V] Ver grabaciones previas")
            print("  [V]olver al menú principal")
            
            choice = input("\nElige opción (G/V/V): ").strip().upper()
            
            if choice == "V":
                # Muestra archivos existentes
                cmd_path = Path(f"data/comandos/{command_name}")
                if cmd_path.exists():
                    files = sorted(cmd_path.glob("grabacion_*.wav"))
                    if files:
                        print(f"\nGrabaciones existentes de '{command_name}':")
                        for i, f in enumerate(files, 1):
                            size_kb = f.stat().st_size / 1024
                            print(f"  {i}. {f.name} ({size_kb:.1f} KB)")
                    else:
                        print(f"\nNo hay grabaciones de '{command_name}' aún")
                else:
                    print(f"\nNo hay grabaciones de '{command_name}' aún")
                continue
            elif choice == "G":
                # Graba nuevo audio
                audio = record_audio(device_id)
                if audio is None:
                    continue
                
                # Normaliza
                audio = normalize_audio(audio)
                
                # Reproduce para verificación
                print("\n¿Quieres escuchar lo grabado antes de guardar?")
                if input("(s/n): ").strip().lower() == 's':
                    preview_audio(audio)
                
                # Confirma antes de guardar
                print(f"\n¿Guardar como: {filename}?")
                if input("(s/n): ").strip().lower() != 's':
                    print("❌ Grabación cancelada - no se guardó")
                    continue
                
                # Guarda
                print(f"\nGuardando como: {filename}")
                if save_audio(audio, command_name, filename):
                    print(f"✓ Grabación #{num} guardada exitosamente")
                    
                    # Pregunta si grabar más del mismo comando
                    if input("\n¿Grabar otro ejemplo de '{}'? (s/n): ".format(command_name)).strip().lower() != 's':
                        break
                else:
                    print("❌ Error al guardar")
            elif choice.upper() == "V":
                break
            else:
                print("❌ Opción inválida")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Grabación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
