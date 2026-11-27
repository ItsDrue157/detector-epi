"""
Parte comum dos scripts de coleta de fotos para re-treino:
    coletar_certo.py   -> fotos de EPI de verdade, rotulado como EPI
    coletar_errado.py  -> fotos do que parece EPI mas não é (boné, colete de futebol)

As fotos são salvas já rotuladas (formato YOLO) e o dados_treino.yaml é refeito
juntando o dataset original com todas as pastas de coleta que existirem.
"""

import time
from datetime import datetime
from pathlib import Path

import cv2

from detectar_epi import AMARELO, VERDE, abrir_captura, categoria_da_classe, texto_com_fundo

DATASET_BASE = "teste"          # dataset original (com train/ e valid/)
YAML_TREINO = "dados_treino.yaml"

# (item, modo) -> pasta das fotos e o que aparece nelas
PASTAS = {
    ("capacete", "errado"): {"pasta": "dados_bone", "o_que": "boné"},
    ("colete", "errado"): {"pasta": "dados_colete_errado", "o_que": "colete de futebol"},
    ("capacete", "certo"): {"pasta": "dados_capacete_certo", "o_que": "capacete de EPI"},
    ("colete", "certo"): {"pasta": "dados_colete_certo", "o_que": "colete de EPI"},
}


def escrever_yaml_treino(nomes):
    """dados_treino.yaml com o dataset original + todas as coletas, para um re-treino
    não fazer o modelo esquecer o que outra coleta ensinou."""
    coletas = [Path(p["pasta"]) for p in PASTAS.values() if (Path(p["pasta"]) / "images").is_dir()]
    linhas = ["# gerado pelos scripts de coleta: dataset original + fotos coletadas",
              "# caminhos relativos a esta pasta", "train:", f"  - {DATASET_BASE}/train/images"]
    linhas += [f"  - {c.as_posix()}/images/train" for c in coletas]
    linhas += ["val:", f"  - {DATASET_BASE}/valid/images"]
    linhas += [f"  - {c.as_posix()}/images/val" for c in coletas]
    linhas += [f"nc: {len(nomes)}", f"names: {[nomes[i] for i in sorted(nomes)]}"]
    Path(YAML_TREINO).write_text("\n".join(linhas) + "\n", encoding="utf-8")


def linha_yolo(caixa, cl, w, h):
    x1, y1, x2, y2 = caixa
    return f"{cl} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def coletar(item, modo, fonte, rotular, intervalo=1.0, tecla_extra=None):
    """Loop da câmera/vídeo comum aos dois scripts.

    rotular(frame, modelo) -> (rotulos, tela, pode_salvar): faz os rótulos YOLO do frame,
    desenha na tela e diz se o frame serve para salvar no modo automático.
    tecla_extra(tecla, frame, modelo) -> rotulos ou None: tecla especial do script (ex.: marcar à mão).
    """
    from ultralytics import YOLO

    if not (Path(DATASET_BASE) / "train" / "images").exists():
        raise SystemExit(f"Não achei {DATASET_BASE}/train/images (dataset original).")
    info = PASTAS[(item, modo)]
    modelo = YOLO("modelos/epi.pt")

    pasta = Path(info["pasta"])
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (pasta / sub).mkdir(parents=True, exist_ok=True)
    escrever_yaml_treino(modelo.names)
    salvas = len(list((pasta / "images").rglob("*.jpg")))

    eh_video = not fonte.isdigit()
    cap = abrir_captura(fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte '{fonte}'.")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    auto = eh_video  # em vídeo já começa salvando sozinho
    ultimo, n_frame = 0.0, 0
    janela = f"Coletar: {info['o_que']}"

    def salvar(frame, rotulos):
        nonlocal salvas
        split = "val" if salvas % 5 == 4 else "train"  # 1 em cada 5 vai para validação
        nome = f"{item}_{modo}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        cv2.imwrite(str(pasta / "images" / split / f"{nome}.jpg"), frame)
        (pasta / "labels" / split / f"{nome}.txt").write_text("\n".join(rotulos), encoding="utf-8")
        salvas += 1

    print(f"\nSalvando em {pasta.resolve()}  (já existem {salvas} fotos)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frame += 1
        rotulos, tela, pode_salvar = rotular(frame, modelo)

        agora = n_frame / fps_video if eh_video else time.time()
        quer_salvar = auto and pode_salvar and agora - ultimo >= intervalo
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("a"):
            auto = not auto
        if tecla == ord(" "):
            quer_salvar = True
        if tecla_extra:
            manuais = tecla_extra(tecla, frame, modelo, janela)
            if manuais is not None:
                salvar(frame, manuais)
                quer_salvar = False

        if quer_salvar:
            ultimo = agora
            salvar(frame, rotulos)
            h, w = frame.shape[:2]
            cv2.rectangle(tela, (0, 0), (w - 1, h - 1), VERDE, 8)

        texto_com_fundo(tela, f"fotos: {salvas} | automatico: {'SIM' if auto else 'nao'} | "
                              "espaco=salvar a=auto q=sair", (10, 30), AMARELO, escala=0.6)
        cv2.imshow(janela, tela)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{salvas} fotos em {pasta.resolve()}")
    print("Agora treine com:")
    print(f"  python treinar.py --dados {YAML_TREINO} --modelo-base modelos/epi.pt --epocas 50")


def classes_do_item(modelo, item):
    return [i for i, n in modelo.names.items() if categoria_da_classe(n) == item]
