# config.py
# Parámetros globales del proyecto (ajusta si necesitas)
import os

# Audio
FS = 44100               # frecuencia de muestreo (Hz) - 44100 Hz es estándar de audio CD
DURATION = 1.0           # duración por grabación (segundos). Asegúrate de que todas las grabaciones tengan el mismo tamaño.
CHANNELS = 1             # mono
TARGET_N = 44100         # tamaño FFT / tamaño de señal (usa potencia de 2 cercano: 44100 ~ 2^15.43, usando 44100)
                         # Si grabas DURATION*FS != TARGET_N, pad/trim en audio_utils.

# FFT / subbands
N_SUBBANDS = 32          # dividir espectro en subbandas

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
COMANDOS_DIR = os.path.join(DATA_DIR, "comandos")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELOS_DIR = os.path.join(DATA_DIR, "modelos")

# Crear carpetas si no existen
for p in (COMANDOS_DIR, TEST_DIR, MODELOS_DIR):
    os.makedirs(p, exist_ok=True)
