"""
Coleta fotos de pessoas de BONÉ para ensinar o modelo que boné não é capacete.

Cada foto salva já vai rotulada: as detecções do modelo atual (colete etc.) viram
o rótulo, MENOS as de capacete, que nessas fotos são bonés. Assim o modelo aprende
que aquilo não é capacete.

IMPORTANTE: só filme pessoas de boné/chapéu/sem nada na cabeça. Se aparecer um
capacete de verdade, ele seria ensinado como "não capacete".

Exemplos:
    python coletar_bones.py                     # webcam
    python coletar_bones.py --fonte bones.mp4   # vídeo gravado (salva 1 foto por segundo)

Teclas: espaço = salvar foto | a = liga/desliga salvar automático | q/ESC = sair

Depois:
    python treinar.py --dados dados_bone/data.yaml --modelo-base modelos/epi_backup.pt --epocas 50
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from detectar_epi import (AMARELO, VERDE, VERMELHO, abrir_captura, categoria_da_classe,
                          texto_com_fundo)


def escrever_yaml(pasta, dataset_base, nomes):
    base = Path(dataset_base).resolve()
    pasta = pasta.resolve()
    linhas = [
        "# gerado pelo coletar_bones.py: dataset original + fotos de boné",
        "train:",
        f"  - {(base / 'train' / 'images').as_posix()}",
        f"  - {(pasta / 'images' / 'train').as_posix()}",
        "val:",
        f"  - {(base / 'valid' / 'images').as_posix()}",
        f"  - {(pasta / 'images' / 'val').as_posix()}",
        f"nc: {len(nomes)}",
        f"names: {[nomes[i] for i in sorted(nomes)]}",
    ]
    (pasta / "data.yaml").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Coleta fotos de boné já rotuladas para re-treino")
    ap.add_argument("--fonte", default="0", help="0/1/2 = webcam, ou caminho de vídeo")
    ap.add_argument("--modelo", default="modelos/epi.pt", help="modelo atual (faz os rótulos de colete etc.)")
    ap.add_argument("--dataset-base", default="teste", help="pasta do dataset original (com train/ e valid/)")
    ap.add_argument("--pasta", default="dados_bone", help="onde salvar as fotos novas")
    ap.add_argument("--intervalo", type=float, default=1.0, help="segundos entre fotos no modo automático")
    ap.add_argument("--conf", type=float, default=0.40, help="confiança mínima dos rótulos de colete etc.")
    ap.add_argument("--largura", type=int, default=1280)
    ap.add_argument("--altura", type=int, default=720)
    args = ap.parse_args()

    if not (Path(args.dataset_base) / "train" / "images").exists():
        sys.exit(f"Não achei {args.dataset_base}/train/images. Use --dataset-base com a pasta do dataset original.")

    from ultralytics import YOLO
    modelo = YOLO(args.modelo)
    nomes = modelo.names
    classes_capacete = {i for i, n in nomes.items() if categoria_da_classe(n) == "capacete"}

    pasta = Path(args.pasta)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (pasta / sub).mkdir(parents=True, exist_ok=True)
    escrever_yaml(pasta, args.dataset_base, nomes)
    salvas = len(list((pasta / "images").rglob("*.jpg")))

    eh_video = not args.fonte.isdigit()
    cap = abrir_captura(args.fonte, args.largura, args.altura)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir a fonte '{args.fonte}'.")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    auto = eh_video  # em vídeo já começa salvando sozinho
    ultimo = 0.0
    n_frame = 0

    print(f"\nSalvando em {pasta.resolve()}  (já existem {salvas} fotos)")
    print("Só filme pessoas de BONÉ. Espaço = salvar | a = automático | q = sair\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frame += 1
        r = modelo.predict(frame, conf=args.conf, verbose=False)[0]
        h, w = frame.shape[:2]
        rotulos, tela = [], frame.copy()
        for (x1, y1, x2, y2), cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            if cl in classes_capacete:  # boné que o modelo achou que era capacete: fica sem rótulo
                cv2.rectangle(tela, p1, p2, VERMELHO, 2)
                cv2.line(tela, p1, p2, VERMELHO, 2)
                texto_com_fundo(tela, "capacete -> descartado", p1, VERMELHO, escala=0.45)
                continue
            cv2.rectangle(tela, p1, p2, VERDE, 2)
            texto_com_fundo(tela, f"{nomes[cl]} (rotulo)", p1, VERDE, escala=0.45)
            rotulos.append(f"{cl} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                           f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")

        agora = n_frame / fps_video if eh_video else time.time()
        salvar = auto and agora - ultimo >= args.intervalo
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("a"):
            auto = not auto
        if tecla == ord(" "):
            salvar = True

        if salvar:
            ultimo = agora
            split = "val" if salvas % 5 == 4 else "train"  # 1 em cada 5 vai para validação
            nome = f"bone_{datetime.now():%Y%m%d_%H%M%S_%f}"
            cv2.imwrite(str(pasta / "images" / split / f"{nome}.jpg"), frame)
            (pasta / "labels" / split / f"{nome}.txt").write_text("\n".join(rotulos), encoding="utf-8")
            salvas += 1
            cv2.rectangle(tela, (0, 0), (w - 1, h - 1), VERDE, 8)

        texto_com_fundo(tela, f"fotos: {salvas} | automatico: {'SIM' if auto else 'nao'} | "
                              "espaco=salvar a=auto q=sair", (10, 30), AMARELO, escala=0.6)
        cv2.imshow("Coletar bones", tela)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{salvas} fotos em {pasta.resolve()}")
    print("Agora treine com:")
    print(f"  python treinar.py --dados {pasta.as_posix()}/data.yaml --modelo-base modelos/epi_backup.pt --epocas 50")


if __name__ == "__main__":
    main()
