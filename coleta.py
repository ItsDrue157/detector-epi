"""
Coleta do COLETE PADRÃO (verde-limão com faixas refletivas) para re-treinar o modelo.

    python coleta.py gravar    # 1. grava vídeos curtos por cenário (frente, costas, aberto...)
                               #    (outra câmera: python coleta.py gravar 1)
    python coleta.py revisar   # 2. passa pelos vídeos novos: o modelo sugere a caixa e você confere

Nenhuma foto entra no dataset sem você ver. Treino e validação são separados POR
VÍDEO (o 2º, 6º, 10º... vídeo de cada cenário vai para a validação), então a
validação nunca tem fotos quase iguais às de treino.

Depois:
    python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detectar_epi import AMARELO, CINZA, VERDE, VERMELHO, abrir_captura, categoria_da_classe, texto_com_fundo

VIDEOS = Path("coleta_colete/videos")    # vídeos gravados (fora do git)
VISTAS = Path("coleta_colete/vistas.txt")  # fotos já revisadas (aceitas ou descartadas)
DADOS = Path("dados_colete")             # dataset final, formato YOLO
MODELO = "modelos/epi.pt"

CENARIOS = [  # (nome, dica). O nome vira o começo do nome de cada vídeo.
    ("frente", "de frente, parado e andando"),
    ("costas", "de costas (o X refletivo)"),
    ("lado", "de lado, os dois lados"),
    ("aberto", "colete aberto, de frente"),
    ("longe", "a 3-5 m da camera"),
    ("bracos", "bracos levantados ou cruzados"),
    ("pouca_luz", "luz fraca ou contra a janela"),
    ("sem_colete", "SEM colete (exemplo negativo)"),
]
NEGATIVO = "sem_colete"  # nunca recebe caixa de colete

INTERVALO = 0.5        # segundos entre uma foto e outra do vídeo
LIMIAR_REPETIDA = 6.0  # diferença média numa miniatura 32x32 abaixo disso = foto repetida
CONF_COLETE = 0.02     # o modelo original acha o colete padrão com só 0.03-0.10
CONF_OUTROS = 0.40     # capacete e outros EPIs que aparecem junto
# O modelo confunde o colete de costas com estes uniformes (amarelo com refletivo).
CONFUNDIDAS = ("C_eletrica", "Operacional")
CONF_CONFUNDIDAS = 0.25


def cenario(video):
    return video.split("__")[0]


def eh_validacao(video):
    """O 2º, 6º, 10º... vídeo de cada cenário vai para a validação (~25%)."""
    mesmos = sorted(v.stem for v in VIDEOS.glob(f"{cenario(video)}__*.mp4"))
    return mesmos.index(video) % 4 == 1


def linha_yolo(caixa, cl, w, h):
    x1, y1, x2, y2 = caixa
    return f"{cl} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def juntar_caixas(caixas):
    """Junta caixas que se sobrepõem ou ficam lado a lado na mesma altura
    (colete aberto: o modelo acha cada metade separada). [x1, y1, x2, y2]."""
    caixas = [list(c) for c in caixas]
    i = 0
    while i < len(caixas):
        for j in range(i + 1, len(caixas)):
            a, b = caixas[i], caixas[j]
            sobre_v = min(a[3], b[3]) - max(a[1], b[1])
            folga_h = max(a[0], b[0]) - min(a[2], b[2])
            if sobre_v >= 0.5 * min(a[3] - a[1], b[3] - b[1]) and folga_h <= 0.25 * (a[2] - a[0] + b[2] - b[0]) / 2:
                caixas[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                del caixas[j]
                break
        else:
            i += 1
    return caixas


def escrever_yaml_treino(nomes):
    """dados_treino.yaml: dataset original + fotos de boné + fotos do colete padrão."""
    extras = [p for p in ("dados_bone", "dados_colete") if Path(p, "images").is_dir()]
    linhas = ["# dataset de treino: original + fotos coletadas", "train:", "  - teste/train/images"]
    linhas += [f"  - {p}/images/train" for p in extras]
    linhas += ["val:", "  - teste/valid/images"]
    linhas += [f"  - {p}/images/val" for p in extras]
    linhas += [f"nc: {len(nomes)}", f"names: {[nomes[i] for i in sorted(nomes)]}"]
    Path("dados_treino.yaml").write_text("\n".join(linhas) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. gravar
# --------------------------------------------------------------------------- #
def gravar(fonte):
    VIDEOS.mkdir(parents=True, exist_ok=True)
    cap = abrir_captura(fonte)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir a fonte '{fonte}'.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 5 <= fps <= 60 else 20
    escolhido, gravador, inicio = 0, None, 0.0
    print("\n1-8 escolhe o cenário | espaço começa/para | q sai")
    print("Grave 20 a 40 s por vídeo, com a pessoa se mexendo. Faça 2 ou mais vídeos por cenário.\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if gravador:
            gravador.write(frame)
        tela = frame.copy()
        y = 30
        for i, (nome, dica) in enumerate(CENARIOS):
            n = len(list(VIDEOS.glob(f"{nome}__*.mp4")))
            cor = VERDE if n >= 2 else AMARELO if n else CINZA
            marca = ">" if i == escolhido else " "
            y += texto_com_fundo(tela, f"{marca}{i + 1} {nome:<11} {n} video(s)  {dica}", (10, y), cor, escala=0.5)
        if gravador:
            texto_com_fundo(tela, f"GRAVANDO {CENARIOS[escolhido][0]}  {time.time() - inicio:.0f}s  (espaco para)",
                            (10, tela.shape[0] - 15), VERMELHO, escala=0.6)
        cv2.imshow("Gravar colete padrao", tela)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if not gravador and ord("1") <= tecla < ord("1") + len(CENARIOS):
            escolhido = tecla - ord("1")
        if tecla == ord(" "):
            if gravador:
                gravador.release()
                gravador = None
                print(f"  vídeo salvo ({time.time() - inicio:.0f} s)")
            else:
                arquivo = VIDEOS / f"{CENARIOS[escolhido][0]}__{datetime.now():%Y%m%d_%H%M%S}.mp4"
                h, w = frame.shape[:2]
                gravador = cv2.VideoWriter(str(arquivo), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                inicio = time.time()
    if gravador:
        gravador.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Próximo passo:  python coleta.py revisar")


# --------------------------------------------------------------------------- #
# 2. revisar
# --------------------------------------------------------------------------- #
def fotos_do_video(caminho):
    """Uma foto a cada INTERVALO segundos, pulando as quase iguais à anterior. Gera (número, foto)."""
    cap = cv2.VideoCapture(str(caminho))
    passo = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) * INTERVALO))
    n, ultima = -1, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % passo:
            continue
        mini = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
        if ultima is not None and np.abs(mini - ultima).mean() < LIMIAR_REPETIDA:
            continue
        ultima = mini
        yield n, frame
    cap.release()


def sugerir(modelo, frame, negativo, cls_colete, cls_confundidas):
    """Caixas sugeridas pelo modelo: (coletes, outros), cada caixa = (x1, y1, x2, y2, classe)."""
    r = modelo.predict(frame, conf=CONF_COLETE, verbose=False)[0]
    coletes, outros = [], []
    for caixa, cl, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
        if cl in cls_colete or (cl in cls_confundidas and cf >= CONF_CONFUNDIDAS):
            coletes.append(caixa)  # C_eletrica/Operacional aqui = o colete padrão de costas
        elif cf >= CONF_OUTROS and cl not in cls_confundidas:
            outros.append((*caixa, cl))
    if negativo:
        coletes = []
    return [(*c, cls_colete[0]) for c in juntar_caixas(coletes)], outros


def revisar():
    videos = sorted(VIDEOS.glob("*.mp4")) if VIDEOS.is_dir() else []
    if not videos:
        sys.exit("Nenhum vídeo gravado. Rode antes:  python coleta.py gravar")
    vistas = set(VISTAS.read_text(encoding="utf-8").split()) if VISTAS.exists() else set()

    from ultralytics import YOLO
    modelo = YOLO(MODELO)
    cls_colete = [i for i, n in modelo.names.items() if categoria_da_classe(n) == "colete"]
    cls_confundidas = [i for i, n in modelo.names.items() if n in CONFUNDIDAS]
    for split in ("train", "val"):
        (DADOS / "images" / split).mkdir(parents=True, exist_ok=True)
        (DADOS / "labels" / split).mkdir(parents=True, exist_ok=True)

    janela = "Revisar colete padrao"
    ajuda = "ENTER=aceitar  r=redesenhar  d=descartar  q=sair"
    print(f"\n{ajuda}")
    print("Caixa boa = justa no colete, do ombro até a barra, sem pegar braço nem fundo.\n")
    aceitas = descartadas = 0
    sair = False
    for video in videos:
        if sair:
            break
        negativo = cenario(video.stem) == NEGATIVO
        split = "val" if eh_validacao(video.stem) else "train"
        for n, frame in fotos_do_video(video):
            nome = f"{video.stem}__{n:06d}"  # começa pelo cenário: sem_colete__... = exemplo negativo
            if nome in vistas:
                continue
            h, w = frame.shape[:2]
            escala = min(1.0, 1280 / w, 720 / h)
            coletes, outros = sugerir(modelo, frame, negativo, cls_colete, cls_confundidas)
            while True:
                tela = cv2.resize(frame, None, fx=escala, fy=escala)
                for x1, y1, x2, y2, cl in coletes:
                    cv2.rectangle(tela, (int(x1 * escala), int(y1 * escala)), (int(x2 * escala), int(y2 * escala)),
                                  VERDE, 3)
                for x1, y1, x2, y2, cl in outros:
                    cv2.rectangle(tela, (int(x1 * escala), int(y1 * escala)), (int(x2 * escala), int(y2 * escala)),
                                  CINZA, 1)
                topo = f"{video.stem} [{split}]" + ("  SEM COLETE: aceite sem caixa" if negativo else "")
                if not coletes and not negativo:
                    topo += "  colete nao achado: aperte r"
                texto_com_fundo(tela, topo, (10, 28), AMARELO, escala=0.6)
                texto_com_fundo(tela, ajuda, (10, tela.shape[0] - 12), AMARELO, escala=0.5)
                cv2.imshow(janela, tela)
                tecla = cv2.waitKey(0) & 0xFF
                if tecla == ord("r") and not negativo:
                    caixas = cv2.selectROIs(janela, cv2.resize(frame, None, fx=escala, fy=escala), showCrosshair=False)
                    if len(caixas):  # ESC sem desenhar nada mantém a sugestão
                        coletes = [(x / escala, y / escala, (x + bw) / escala, (y + bh) / escala, cls_colete[0])
                                   for x, y, bw, bh in caixas]
                elif tecla in (13, 10, ord(" ")):  # aceitar
                    cv2.imwrite(str(DADOS / "images" / split / f"{nome}.jpg"), frame)
                    (DADOS / "labels" / split / f"{nome}.txt").write_text(
                        "\n".join(linha_yolo(c[:4], c[4], w, h) for c in coletes + outros), encoding="utf-8")
                    aceitas += 1
                    break
                elif tecla == ord("d"):
                    descartadas += 1
                    break
                elif tecla in (ord("q"), 27):
                    sair = True
                    break
            if sair:
                break
            vistas.add(nome)
            with VISTAS.open("a", encoding="utf-8") as f:
                f.write(nome + "\n")

    cv2.destroyAllWindows()
    escrever_yaml_treino(modelo.names)
    total = {s: len(list((DADOS / "images" / s).glob("*.jpg"))) for s in ("train", "val")}
    print(f"\nNesta sessão: {aceitas} aceitas, {descartadas} descartadas.")
    print(f"dados_colete/: {total['train']} fotos de treino, {total['val']} de validação. dados_treino.yaml atualizado.")
    if not sair:
        print("Todos os vídeos revisados. Treine com:")
        print("  python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50")


if __name__ == "__main__":
    etapa = sys.argv[1] if len(sys.argv) > 1 else ""
    if etapa == "gravar":
        gravar(sys.argv[2] if len(sys.argv) > 2 else "0")
    elif etapa == "revisar":
        revisar()
    else:
        sys.exit(__doc__)
