import os
import sys

# Ensure repo root on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from recognition import recognizer


def find_wavs_in_test():
    test_dir = os.path.join(REPO_ROOT, 'data', 'test')
    if not os.path.isdir(test_dir):
        return []
    return [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith('.wav')]


def find_wavs_in_comandos():
    comandos_dir = os.path.join(REPO_ROOT, 'data', 'comandos')
    if not os.path.isdir(comandos_dir):
        return []
    wavs = []
    for sub in os.listdir(comandos_dir):
        subdir = os.path.join(comandos_dir, sub)
        if not os.path.isdir(subdir):
            continue
        for f in os.listdir(subdir):
            if f.lower().endswith('.wav'):
                wavs.append(os.path.join(subdir, f))
    return wavs


def main():
    wavs = find_wavs_in_test()
    source = 'data/test/'
    if not wavs:
        wavs = find_wavs_in_comandos()
        source = 'data/comandos/*/'

    if not wavs:
        print('No se encontraron archivos .wav en data/test ni en data/comandos/.')
        return

    print(f'Encontrados {len(wavs)} WAVs en {source}. Ejecutando reconocimiento...')
    results = []
    for w in sorted(wavs):
        try:
            print('\n-> Archivo:', w)
            cmd = recognizer.recognize_from_file(w)
            results.append((w, cmd))
        except Exception as e:
            print(f'Error procesando {w}: {e}')
            results.append((w, None))

    print('\n=== Resumen ===')
    for w, cmd in results:
        print(f'{os.path.relpath(w, REPO_ROOT):40} -> {cmd}')


if __name__ == '__main__':
    main()
