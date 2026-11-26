"""
Baixa um modelo de EPI já treinado (sem precisar treinar) e salva em modelos/epi.pt.

Modelo: SafetyVision YOLOv8s (v2) - https://huggingface.co/ayushgupta7777/safetyvision-yolov8
  - treinado em ~80 mil imagens de canteiros/indústria
  - classes: Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person,
             Gloves, Goggles, Mask, ... (13 no total)
  - licença AGPL-3.0

Uso:
    python baixar_modelo.py
    python detectar_epi.py
"""

import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/ayushgupta7777/safetyvision-yolov8/resolve/main/v2/best.pt"
DESTINO = Path("modelos") / "epi.pt"


def progresso(blocos, tam_bloco, total):
    if total > 0:
        feito = min(blocos * tam_bloco, total)
        pct = feito * 100 / total
        barra = "#" * int(pct / 4)
        sys.stdout.write(f"\r  [{barra:<25}] {pct:5.1f}%  {feito / 1e6:5.1f}/{total / 1e6:.1f} MB")
        sys.stdout.flush()


def main():
    if DESTINO.exists() and "--forcar" not in sys.argv:
        print(f"{DESTINO} já existe. Use --forcar para baixar de novo.")
        return
    DESTINO.parent.mkdir(exist_ok=True)
    tmp = DESTINO.with_suffix(".baixando")
    print(f"Baixando modelo pronto de EPI...\n  {URL}")
    try:
        urllib.request.urlretrieve(URL, tmp, progresso)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"\nFalha no download: {e}\n"
                 f"Baixe manualmente pelo navegador e salve como {DESTINO}:\n  {URL}")
    tmp.replace(DESTINO)
    print(f"\nPronto! Modelo salvo em {DESTINO.resolve()}")

    try:
        from ultralytics import YOLO
        nomes = YOLO(str(DESTINO)).names
        print("Classes:", ", ".join(nomes.values()))
    except Exception:
        pass
    print("\nAgora rode:  python detectar_epi.py")


if __name__ == "__main__":
    main()
