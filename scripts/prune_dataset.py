#!/usr/bin/env python3
"""
prune_dataset.py

Elimina (o mueve a la papelera) un porcentaje de archivos de audio
bajo `data/comandos`.

Uso:
    python scripts/prune_dataset.py [--percent P] [--mode MODE] [--dry-run] [--seed S] [--trash-dir DIR]

Opciones:
  --percent P      Porcentaje de eliminación (0-100). Por defecto 30.
  --mode MODE      "per-folder" (por defecto) o "global". "per-folder" elimina P% de cada carpeta.
                   "global" elimina P% del total distribuidos aleatoriamente.
  --dry-run        No borra ni mueve nada, sólo muestra lo que se haría.
  --seed S         Semilla aleatoria para reproducibilidad.
  --trash-dir DIR  Si se especifica (por defecto data/trash), los archivos se moverán allí
                   en lugar de borrarse permanentemente.

Ejemplos:
  python scripts/prune_dataset.py --percent 30 --mode per-folder
  python scripts/prune_dataset.py --percent 25 --mode global --dry-run --seed 42

"""

import argparse
import os
import random
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
COMANDOS_DIR = ROOT / 'data' / 'comandos'

AUDIO_EXTS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}


def list_audio_files(folder: Path):
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def ensure_trash(trash_dir: Path):
    trash_dir.mkdir(parents=True, exist_ok=True)


def move_to_trash(src: Path, trash_dir: Path):
    ensure_trash(trash_dir)
    # Keep folder structure to avoid name collisions
    rel = src.relative_to(COMANDOS_DIR)
    dest = trash_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def delete_file(path: Path):
    try:
        path.unlink()
    except Exception as e:
        print(f"Error borrando {path}: {e}")


def prune_per_folder(percent, dry_run, trash_dir, do_delete):
    total = 0
    removed = 0
    for folder in sorted(COMANDOS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        files = list_audio_files(folder)
        n = len(files)
        if n == 0:
            continue
        k = max(1, int(round(n * percent / 100.0)))
        chosen = random.sample(files, k)
        print(f"Folder: {folder.name} - {n} files, removing {k}")
        for p in chosen:
            total += 1
            if dry_run:
                print(f" DRY: would remove {p}")
            else:
                if do_delete:
                    delete_file(p)
                    print(f" Deleted {p}")
                else:
                    move_to_trash(p, trash_dir)
                    print(f" Moved to trash: {p}")
            removed += 1
    print(f"\nSummary: processed {total} files selected for removal ({removed} acted on)")


def prune_global(percent, dry_run, trash_dir, do_delete):
    all_files = []
    for folder in sorted(COMANDOS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        all_files.extend(list_audio_files(folder))
    total_files = len(all_files)
    if total_files == 0:
        print("No audio files found under data/comandos")
        return
    k = max(1, int(round(total_files * percent / 100.0)))
    chosen = random.sample(all_files, k)
    print(f"Global: {total_files} files total, removing {k}")
    for p in chosen:
        if dry_run:
            print(f" DRY: would remove {p}")
        else:
            if do_delete:
                delete_file(p)
                print(f" Deleted {p}")
            else:
                move_to_trash(p, trash_dir)
                print(f" Moved to trash: {p}")
    print(f"\nSummary: selected {k} files for removal (dry_run={dry_run})")


def parse_args():
    parser = argparse.ArgumentParser(description='Prune audio files under data/comandos')
    parser.add_argument('--percent', type=float, default=30.0, help='Porcentaje a eliminar (0-100)')
    parser.add_argument('--mode', choices=['per-folder', 'global'], default='per-folder', help='Modo de eliminación')
    parser.add_argument('--dry-run', action='store_true', help='Mostrar acciones sin ejecutar')
    parser.add_argument('--seed', type=int, default=None, help='Semilla aleatoria')
    parser.add_argument('--trash-dir', type=str, default=str(ROOT / 'data' / 'trash'), help='Directorio donde mover archivos en vez de borrarlos')
    parser.add_argument('--delete', action='store_true', help='Borrar permanentemente en vez de mover a trash')
    return parser.parse_args()


def main():
    args = parse_args()
    if not COMANDOS_DIR.exists():
        print(f"No existe {COMANDOS_DIR}. Asegúrate de ejecutar desde el directorio del proyecto.")
        return
    if args.percent <= 0:
        print("Percent must be > 0")
        return
    if args.percent >= 100:
        print("Percent must be < 100")
        return
    if args.seed is not None:
        random.seed(args.seed)

    trash_dir = Path(args.trash_dir)
    print(f"Running prune: percent={args.percent}, mode={args.mode}, dry_run={args.dry_run}, delete={args.delete}")
    if args.mode == 'per-folder':
        prune_per_folder(args.percent, args.dry_run, trash_dir, args.delete)
    else:
        prune_global(args.percent, args.dry_run, trash_dir, args.delete)


if __name__ == '__main__':
    main()
