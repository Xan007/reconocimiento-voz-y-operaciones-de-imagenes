import os
import uuid
import soundfile as sf

# Carpetas donde están tus grabaciones (puedes modificar o generar dinámicamente)
folders = [
    "data/comandos/uno",
    "data/comandos/dos",
]

# Duración mínima en segundos
MIN_DURATION = 1.0


def is_wav(fname: str) -> bool:
    return fname.lower().endswith('.wav')


for folder in folders:
    print(f"\nProcesando carpeta: {folder}")
    if not os.path.isdir(folder):
        print(f"⚠️  No existe la carpeta {folder}, saltando...")
        continue

    # Listar archivos .wav reales en la carpeta (no modificar los nombres aquí)
    try:
        all_files = [f for f in os.listdir(folder) if is_wav(f)]
    except Exception as e:
        print(f"Error al listar {folder}: {e}")
        continue

    valid = []  # tuples: (orig_name, orig_path, mtime)
    for fname in all_files:
        orig_path = os.path.join(folder, fname)
        try:
            data, fs = sf.read(orig_path)
            dur = len(data) / fs if fs else 0
            if dur >= MIN_DURATION:
                try:
                    mtime = os.path.getmtime(orig_path)
                except Exception:
                    mtime = 0
                valid.append((fname, orig_path, mtime))
            else:
                # Eliminar archivos que no cumplan duración mínima
                try:
                    os.remove(orig_path)
                    print(f"🗑️  {fname} eliminado ({dur:.2f} s < {MIN_DURATION}s)")
                except Exception as e:
                    print(f"Error eliminando {fname}: {e}")
        except Exception as e:
            print(f"Error al leer {fname}: {e}")

    print(f"✅ {len(valid)} archivos válidos encontrados.")
    if not valid:
        continue

    # Paso 1: renombrar TODO a nombres temporales aleatorios para evitar colisiones
    temp_items = []  # tuples: (temp_name, temp_path, mtime)
    for fname, orig_path, mtime in valid:
        ext = os.path.splitext(fname)[1]
        temp_name = f"tmp_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(folder, temp_name)
        try:
            os.rename(orig_path, temp_path)
            temp_items.append((temp_name, temp_path, mtime))
            print(f"→ {fname}  →  {temp_name}")
        except Exception as e:
            print(f"Error renombrando {fname} a temporal: {e}")

    # Paso 2: renombrar temporales a formato final COMMAND_XX (ordenado por fecha de modificación)
    # Ordenar por mtime para mantener orden cronológico (más estable que ordenar por nombre)
    temp_items.sort(key=lambda t: t[2])

    for i, (temp_name, temp_path, _) in enumerate(temp_items, start=1):
        ext = os.path.splitext(temp_name)[1]
        base_name = os.path.basename(folder.rstrip(os.sep))
        final_name = f"{base_name}_{i:02d}{ext}"
        final_path = os.path.join(folder, final_name)

        # Evitar sobrescribir: si ya existe, añadir sufijo incremental
        suffix = 0
        while os.path.exists(final_path):
            suffix += 1
            final_name = f"{base_name}_{i:02d}_{suffix}{ext}"
            final_path = os.path.join(folder, final_name)

        try:
            os.rename(temp_path, final_path)
            print(f"→ {temp_name}  →  {final_name}")
        except Exception as e:
            print(f"Error renombrando temporal {temp_name} a final: {e}")

print("\n✔️  Proceso completado.")
