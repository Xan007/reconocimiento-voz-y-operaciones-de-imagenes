# main.py
import os
from queue import Queue
import numpy as np
import sounddevice as sd
from training.trainer import train_all, train_command
from recognition.recognizer import recognize_from_mic, recognize_from_file
from config import COMANDOS_DIR, TEST_DIR, FS, TARGET_N
from processing.audio_utils import normalize_audio, pad_or_trim
from processing.fft_utils import analyze_signal

def audio_callback(indata, frames, time, status):
    """Callback para procesar el audio en tiempo real."""
    if status:
        print('Error:', status)
    # Convertir a mono si es estéreo
    if indata.shape[1] > 1:
        data = np.mean(indata, axis=1)
    else:
        data = indata[:, 0]
    # Añadir los nuevos datos al buffer circular
    audio_callback.buffer[:-frames] = audio_callback.buffer[frames:]
    audio_callback.buffer[-frames:] = data
    # Procesar si tenemos suficientes datos y no estamos ya procesando
    if not audio_callback.processing:
        audio_callback.processing = True
        try:
            # Normalizar y preparar el audio
            x = normalize_audio(audio_callback.buffer.copy())
            x = pad_or_trim(x)
            # Analizar y obtener energías
            _, _, energies = analyze_signal(x)
            # Añadir a la cola para procesar en el hilo principal
            audio_callback.queue.put(energies)
        finally:
            audio_callback.processing = False


def continuous_recognition():
    """
    Reconocimiento continuo desde el micrófono usando una ventana deslizante.
    Detecta comandos en tiempo real sin esperas fijas.
    """
    print("🎙️ Reconocimiento continuo iniciado. Presiona Ctrl+C para detener.\n")
    
    # Configurar el callback con su estado
    audio_callback.buffer = np.zeros(TARGET_N, dtype=np.float32)
    audio_callback.processing = False
    audio_callback.queue = Queue()
    
    # Configurar stream de audio
    stream = sd.InputStream(
        channels=1,
        samplerate=FS,
        blocksize=int(FS * 0.4),  # procesar cada 100ms
        callback=audio_callback
    )
    
    try:
        with stream:
            print("Escuchando... (Ctrl+C para detener)")
            from recognition import recognizer
            models = recognizer.load_models()
            
            while True:
                # Esperar nuevos datos de audio (timeout para poder interrumpir)
                try:
                    energies = audio_callback.queue.get(timeout=0.1)
                    cmd, diffs, is_valid = recognizer.compare_with_models(energies, models)
                    
                    #if cmd is not None:
                        #print(f"👉 Detectado: {cmd.upper()}")
                except Exception as e:
                    continue
    
    except KeyboardInterrupt:
        print("\n🛑 Reconocimiento detenido.")
    finally:
        # Limpiar cola
        while True:
            try:
                audio_callback.queue.get_nowait()
            except:
                break


def main():
    while True:
        print("\n=== Reconocimiento de comandos por FFT ===")
        print("[1] Entrenar todos los comandos")
        print("[2] Entrenar un comando específico")
        print("[3] Reconocimiento continuo (micrófono)")
        print("[4] Reconocer desde archivo WAV")
        print("[5] Salir")

        opcion = input("Seleccione una opción: ").strip()

        if opcion == "1":
            train_all()

        elif opcion == "2":
            print("Carpetas disponibles en data/comandos/:")
            for d in os.listdir(COMANDOS_DIR):
                print(" -", d)
            cmd = input("Nombre del comando a entrenar: ").strip()
            if cmd:
                train_command(cmd)

        elif opcion == "3":
            continuous_recognition()

        elif opcion == "4":
            print("Archivos disponibles en data/test/:")
            for f in os.listdir(TEST_DIR):
                if f.endswith(".wav"):
                    print(" -", f)
            filename = input("Archivo (sin ruta): ").strip()
            path = os.path.join(TEST_DIR, filename)
            if os.path.isfile(path):
                recognize_from_file(path)
            else:
                print("No se encontró el archivo.")

        elif opcion == "5":
            print("Saliendo...")
            break

        else:
            print("Opción inválida. Intente de nuevo.")


if __name__ == "__main__":
    main()
