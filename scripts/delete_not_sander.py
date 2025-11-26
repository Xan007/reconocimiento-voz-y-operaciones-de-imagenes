#!/usr/bin/env python3
"""
delete_not_sander.py

Busca y elimina (o mueve a trash) todos los archivos de audio
bajo `data/comandos` cuyo nombre (sin extensión) NO contiene "sander"
(case-insensitive).

Opciones:
  --dry-run       Mostrar archivos que serían eliminados (por defecto True)
  --delete        Borrar permanentemente en lugar de mover a trash
  --yes           Confirmar sin preguntar
  --trash-dir DIR Directorio donde mover archivos (por defecto data/trash/not_sander_removed_<timestamp>)
  --folder PATH   Limitar a una subcarpeta de data/comandos

Ejemplos:
  python scripts/delete_not_sander.py --dry-run
  python scripts/delete_not_sander.py --trash-dir data/trash/not_sander --yes
  python scripts/delete_not_sander.py --delete --yes

IMPORTANTE: Por seguridad el comportamiento por defecto es "--dry-run". Usa --yes para confirmar.
"""

from pathlib import Path
import argparse
import shutil
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
COMANDOS_DIR = ROOT / 'data' / 'comandos'
AUDIO_EXTS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
PATTERN = 'sander'


def list_audio_files(base_dir: Path, subfolder: Path = None):
    files = []
    search_root = base_dir if subfolder is None else (base_dir / subfolder)
    if not search_root.exists():
        return files
    for p in search_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def ensure_trash(trash_dir: Path):
    trash_dir.mkdir(parents=True, exist_ok=True)


def move_to_trash(src: Path, trash_dir: Path):
    ensure_trash(trash_dir)
    rel = src.relative_to(COMANDOS_DIR)
    dest = trash_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def delete_file(path: Path):
    try:
        path.unlink()
    except Exception as e:
        print(f"Error borrando {path}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Delete audio files NOT containing 'sander' under data/comandos")
    p.add_argument('--dry-run', action='store_true', default=False, help='Mostrar lo que se haría y no tocar archivos')
    p.add_argument('--delete', action='store_true', default=False, help='Borrar permanentemente en vez de mover a trash')
    p.add_argument('--yes', action='store_true', default=False, help='Confirmar sin preguntar')
    p.add_argument('--trash-dir', type=str, default=None, help='Directorio donde mover archivos (por defecto data/trash/not_sander_removed_<ts>)')
    p.add_argument('--folder', type=str, default=None, help='Subcarpeta dentro de data/comandos para limitar la búsqueda')
    return p.parse_args()


def main():
    args = parse_args()

    if not COMANDOS_DIR.exists():
        print(f"No existe {COMANDOS_DIR}. Ejecuta desde la raíz del proyecto.")
        sys.exit(1)

    subfolder = None
    if args.folder:
        subfolder = Path(args.folder)
        if not (COMANDOS_DIR / subfolder).exists():
            print(f"La subcarpeta {subfolder} no existe dentro de data/comandos")
            sys.exit(1)

    all_files = list_audio_files(COMANDOS_DIR, subfolder)
    if not all_files:
        print("No se encontraron archivos de audio bajo data/comandos")
        return

    files_to_remove = [p for p in all_files if PATTERN.lower() not in p.stem.lower()]

    if not files_to_remove:
        print(f"No se encontraron archivos que NO contengan '{PATTERN}' en su nombre.")
        return

    print(f"Encontrados {len(files_to_remove)} archivos que NO contienen '{PATTERN}':")
    for f in files_to_remove:
        print(' -', f)

    if args.dry_run:
        print('\nDRY-RUN activado, no se realizarán cambios.')
        return

    if not args.yes:
        reply = input(f"¿Deseas proceder a {'borrar' if args.delete else 'mover a trash'} estos {len(files_to_remove)} archivos? [y/N]: ")
        if reply.strip().lower() not in ('y', 'yes'):
            print('Operación cancelada por el usuario.')
            return

    # preparar trash dir
    if args.trash_dir:
        trash_dir = Path(args.trash_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        trash_dir = ROOT / 'data' / 'trash' / f'not_sander_removed_{ts}'

    if args.delete:
        for f in files_to_remove:
            delete_file(f)
            print('Deleted', f)
    else:
        for f in files_to_remove:
            move_to_trash(f, trash_dir)
            print('Moved to trash:', f)

    print('\nOperación completada.')

if __name__ == '__main__':
    main()
