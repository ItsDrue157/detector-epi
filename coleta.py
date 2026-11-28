"""
Coleta do COLETE PADRÃO (verde-limão com faixas refletivas) em etapas, para re-treinar o modelo.

Cada etapa é separada e pode ser repetida. Nenhuma foto entra no dataset sem
passar pela revisão. Treino e validação são separados POR VÍDEO, então a
validação nunca tem fotos quase iguais às de treino.

    python coleta.py gravar      # 1. grava vídeos curtos por cenário (frente, costas, aberto...)
    python coleta.py extrair     # 2. tira fotos dos vídeos, sem borradas nem repetidas
    python coleta.py rotular     # 3. o modelo atual sugere as caixas (offline)
    python coleta.py revisar     # 4. você confere foto por foto: aceita, corrige ou descarta
    python coleta.py montar      # 5. gera dados_colete/ só com as aceitas e o dados_treino.yaml
    python coleta.py status      # resumo de todas as etapas

Roda na CPU. Com --gpu (NVIDIA RTX série 40 ou 50), o "rotular" fica mais rápido
e usa imagem maior:
    python coleta.py rotular --gpu

Depois de montar:
    python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50

Ciclo recomendado: grave tudo, revise um primeiro lote, monte e treine. O treino
troca o modelos/epi.pt pelo novo, que sugere caixas melhores. Então rode
"rotular --refazer" (só refaz as fotos ainda não revisadas) e continue revisando.
"""

import argparse
import ast
import csv
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detectar_epi import (AMARELO, CINZA, VERDE, VERMELHO, abrir_captura, categoria_da_classe,
                          escolher_perfil, texto_com_fundo)

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
TRABALHO = Path("coleta_colete")       # vídeos, fotos e sugestões (fora do git)
VIDEOS, FOTOS, SUGERIDOS, REVISADOS = (TRABALHO / p for p in ("videos", "fotos", "sugeridos", "revisados"))
REVISAO_CSV = TRABALHO / "revisao.csv"
DADOS = Path("dados_colete")           # dataset final (formato YOLO), gerado pelo "montar"
MODELO = "modelos/epi.pt"
DATA_ORIGINAL = Path("teste/data.yaml")

# Cenários a gravar. Mude à vontade; o nome vira o começo do nome de cada vídeo.
CENARIOS = [
    ("frente", "de frente, parado e andando"),
    ("costas", "de costas (o X refletivo)"),
    ("lado", "de lado, os dois lados"),
    ("aberto", "colete aberto, de frente"),
    ("longe", "a 3-5 m da camera"),
    ("bracos", "bracos levantados ou cruzados"),
    ("pouca_luz", "luz fraca ou contra a janela"),
    ("sem_colete", "SEM colete (exemplo negativo)"),
]
NEGATIVO = "sem_colete"  # cenário sem colete: nunca recebe caixa de colete

FOTOS_POR_SEGUNDO = 2      # amostragem do "extrair"
MAX_FOTOS_POR_VIDEO = 60
BORRADO_RELATIVO = 0.4     # nitidez abaixo de 40% da mediana do próprio vídeo = foto borrada
LIMIAR_REPETIDO = 6.0      # diferença média (0-255) numa miniatura 32x32 abaixo disso = repetida

CONF_COLETE = 0.02         # o modelo atual acha o colete padrão com 0.03-0.10: aceita bem baixo
CONF_CERTA = 0.40          # acima disso a sugestão vem verde; abaixo, amarela (duvidosa)
CONF_OUTROS = 0.40         # capacete e outros EPIs que aparecem junto
# Uniformes que o modelo confunde com o colete padrão de costas (amarelo com refletivo).
CONFUNDIDAS_COM_COLETE = ("C_eletrica", "Operacional")
CONF_CONFUNDIDAS = 0.25
FRACAO_VAL = 0.20          # fração dos vídeos (por cenário) que vai para a validação
SEMENTE = 42


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def cenario_do_video(nome):
    return nome.split("__")[0]


def videos():
    return sorted(VIDEOS.glob("*.mp4")) if VIDEOS.is_dir() else []


def fotos_extraidas():
    """[(video, foto)] de todas as fotos extraídas."""
    if not FOTOS.is_dir():
        return []
    return [(d.name, p.stem) for d in sorted(FOTOS.iterdir()) if d.is_dir() for p in sorted(d.glob("*.jpg"))]


def ler_revisao():
    if not REVISAO_CSV.exists():
        return {}
    with REVISAO_CSV.open(encoding="utf-8", newline="") as f:
        return {(l["video"], l["foto"]): l for l in csv.DictReader(f)}


def gravar_revisao(revisao):
    REVISAO_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = REVISAO_CSV.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "foto", "status", "corrigida", "data"])
        w.writeheader()
        for k in sorted(revisao):
            w.writerow(revisao[k])
    tmp.replace(REVISAO_CSV)


def ler_sugestao(video, foto):
    p = SUGERIDOS / video / f"{foto}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def linha_yolo(caixa, cl, w, h):
    x1, y1, x2, y2 = caixa
    return f"{cl} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def ler_yolo(caminho, w, h):
    """Rótulos YOLO de um .txt -> [{"cls", "xyxy"}] em pixels."""
    caixas = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        p = linha.split()
        if len(p) != 5:
            continue
        cl, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        caixas.append({"cls": cl, "xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                                           (cy + bh / 2) * h]})
    return caixas


def nomes_das_classes():
    """Nomes das classes, lidos do data.yaml do dataset original (linha "names: [...]")."""
    for linha in DATA_ORIGINAL.read_text(encoding="utf-8").splitlines():
        if linha.startswith("names:"):
            return dict(enumerate(ast.literal_eval(linha.split(":", 1)[1].strip())))
    sys.exit(f"Não achei 'names:' em {DATA_ORIGINAL}.")


def escrever_yaml_treino(nomes):
    """dados_treino.yaml: dataset original + fotos de boné + fotos do colete padrão."""
    extras = [p for p in ("dados_bone", "dados_colete") if Path(p, "images").is_dir()]
    linhas = ["# dataset de treino: original + fotos coletadas", "train:", "  - teste/train/images"]
    linhas += [f"  - {p}/images/train" for p in extras]
    linhas += ["val:", "  - teste/valid/images"]
    linhas += [f"  - {p}/images/val" for p in extras]
    linhas += [f"nc: {len(nomes)}", f"names: {[nomes[i] for i in sorted(nomes)]}"]
    Path("dados_treino.yaml").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def juntar_caixas(caixas):
    """Junta caixas de colete que se sobrepõem ou ficam lado a lado na mesma altura
    (colete aberto: o modelo acha cada metade separada). [(x1, y1, x2, y2, conf, fonte)]."""
    caixas = [list(c) for c in caixas]
    juntou = True
    while juntou:
        juntou = False
        for i in range(len(caixas)):
            for j in range(i + 1, len(caixas)):
                a, b = caixas[i], caixas[j]
                sobreposicao_v = min(a[3], b[3]) - max(a[1], b[1])
                menor_altura = min(a[3] - a[1], b[3] - b[1])
                folga_h = max(a[0], b[0]) - min(a[2], b[2])  # negativa = se sobrepõem
                largura_media = (a[2] - a[0] + b[2] - b[0]) / 2
                if sobreposicao_v >= 0.5 * menor_altura and folga_h <= 0.25 * largura_media:
                    forte = a if a[4] >= b[4] else b
                    caixas[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]),
                                 forte[4], "+".join(sorted({a[5], b[5]}))]
                    del caixas[j]
                    juntou = True
                    break
            if juntou:
                break
    return caixas


def barra(i, total, inicio, texto=""):
    feito = int(30 * i / max(total, 1))
    passado = time.time() - inicio
    resta = passado / max(i, 1) * (total - i)
    print(f"\r  [{'#' * feito}{'.' * (30 - feito)}] {i}/{total}  {texto}  falta ~{resta:4.0f}s ",
          end="", flush=True)


# --------------------------------------------------------------------------- #
# 1. gravar
# --------------------------------------------------------------------------- #
def cmd_gravar(args):
    VIDEOS.mkdir(parents=True, exist_ok=True)
    cap = abrir_captura(args.fonte)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir a fonte '{args.fonte}'.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 5 <= fps <= 60 else 20
    janela = "Gravar colete padrao"
    escolhido, gravador, inicio, arquivo = 0, None, 0.0, None

    def contagem():
        n = {c: 0 for c, _ in CENARIOS}
        for v in videos():
            n[cenario_do_video(v.stem)] = n.get(cenario_do_video(v.stem), 0) + 1
        return n

    n = contagem()
    print("\n1-8 escolhe o cenário | espaço começa/para | q sai")
    print("Grave 20 a 40 s por vídeo, com a pessoa se mexendo. Vários vídeos curtos > um longo.\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if gravador is not None:
            gravador.write(frame)
        tela = frame.copy()
        y = 30
        for i, (nome, dica) in enumerate(CENARIOS):
            marca = ">" if i == escolhido else " "
            cor = VERDE if n.get(nome, 0) >= 2 else (AMARELO if n.get(nome, 0) else CINZA)
            y += texto_com_fundo(tela, f"{marca}{i + 1} {nome:<11} {n.get(nome, 0)} video(s)  {dica}",
                                 (10, y), cor, escala=0.5)
        h = tela.shape[0]
        if gravador is not None:
            cv2.circle(tela, (tela.shape[1] - 30, 30), 12, VERMELHO, -1)
            texto_com_fundo(tela, f"GRAVANDO {CENARIOS[escolhido][0]}  {time.time() - inicio:4.0f}s  "
                                  "(espaco para parar)", (10, h - 15), VERMELHO, escala=0.6)
        else:
            texto_com_fundo(tela, "espaco = gravar | 1-8 = cenario | q = sair  (verde = 2+ videos)",
                            (10, h - 15), AMARELO, escala=0.6)
        cv2.imshow(janela, tela)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if gravador is None and ord("1") <= tecla < ord("1") + len(CENARIOS):
            escolhido = tecla - ord("1")
        if tecla == ord(" "):
            if gravador is None:
                arquivo = VIDEOS / f"{CENARIOS[escolhido][0]}__{datetime.now():%Y%m%d_%H%M%S}.mp4"
                hh, ww = frame.shape[:2]
                gravador = cv2.VideoWriter(str(arquivo), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
                inicio = time.time()
            else:
                gravador.release()
                gravador = None
                print(f"  salvo {arquivo.name} ({time.time() - inicio:.0f} s)")
                n = contagem()

    if gravador is not None:
        gravador.release()
        print(f"  salvo {arquivo.name} ({time.time() - inicio:.0f} s)")
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{len(videos())} vídeos em {VIDEOS.resolve()}")
    print("Próximo passo:  python coleta.py extrair")


# --------------------------------------------------------------------------- #
# 2. extrair
# --------------------------------------------------------------------------- #
def cmd_extrair(args):
    pendentes = [v for v in videos() if not (FOTOS / v.stem / ".extraido").exists()]
    if not pendentes:
        print("Nenhum vídeo novo para extrair. Grave com:  python coleta.py gravar")
        return
    for v in pendentes:
        destino = FOTOS / v.stem
        if destino.exists():
            shutil.rmtree(destino)  # extração interrompida no meio: começa de novo
        destino.mkdir(parents=True)
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        passo = max(1, round(fps / FOTOS_POR_SEGUNDO))
        # 1ª passada: nitidez de cada foto amostrada. O limite é relativo ao próprio vídeo,
        # porque cada câmera tem uma nitidez "normal" diferente.
        nitidez, n = {}, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % passo == 0:
                nitidez[n] = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            n += 1
        limite = BORRADO_RELATIVO * float(np.median(list(nitidez.values()))) if nitidez else 0
        # 2ª passada: salva as nítidas e não repetidas
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        n, salvas, borradas, repetidas, ultima = -1, [], 0, 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n not in nitidez:
                continue
            if nitidez[n] < limite:
                borradas += 1
                continue
            cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(cinza, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
            if ultima is not None and np.abs(mini - ultima).mean() < LIMIAR_REPETIDO:
                repetidas += 1
                continue
            ultima = mini
            p = destino / f"{n:06d}.jpg"
            cv2.imwrite(str(p), frame)
            salvas.append(p)
        cap.release()
        excesso = 0
        if len(salvas) > MAX_FOTOS_POR_VIDEO:  # mantém fotos espalhadas pelo vídeo todo
            manter = set(np.linspace(0, len(salvas) - 1, MAX_FOTOS_POR_VIDEO).round().astype(int))
            for i, p in enumerate(salvas):
                if i not in manter:
                    p.unlink()
                    excesso += 1
        ficou = len(salvas) - excesso
        (destino / ".extraido").write_text(json.dumps({"fotos": ficou, "borradas": borradas,
                                                       "repetidas": repetidas, "excesso": excesso}),
                                           encoding="utf-8")
        print(f"  {v.name}: {ficou} fotos  (descartadas: {borradas} borradas, {repetidas} repetidas, "
              f"{excesso} acima do limite)")
    print("Próximo passo:  python coleta.py rotular   (ou  rotular --gpu)")


# --------------------------------------------------------------------------- #
# 3. rotular (sugestões)
# --------------------------------------------------------------------------- #
def sugerir(modelo, frame, cenario, cls_colete, cls_confundidas, dispositivo, imgsz):
    h, w = frame.shape[:2]
    r = modelo.predict(frame, conf=CONF_COLETE, imgsz=imgsz, device=dispositivo, verbose=False)[0]
    coletes, outros = [], []
    for caixa, cl, cf in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.cls.cpu().numpy().astype(int),
                             r.boxes.conf.cpu().numpy()):
        cf = float(cf)
        if cl in cls_colete or (cl in cls_confundidas and cf >= CONF_CONFUNDIDAS):
            coletes.append((*caixa, cf, modelo.names[int(cl)]))
        elif cf >= CONF_OUTROS and cl not in cls_confundidas:
            outros.append({"cls": int(cl), "xyxy": caixa, "conf": cf, "fonte": modelo.names[int(cl)],
                           "colete": False, "duvidosa": False})
    caixas = []
    if cenario != NEGATIVO:
        for x1, y1, x2, y2, cf, fonte in juntar_caixas(coletes):
            confundida = any(n in fonte for n in CONFUNDIDAS_COM_COLETE)
            caixas.append({"cls": cls_colete[0], "xyxy": [x1, y1, x2, y2], "conf": cf, "fonte": fonte,
                           "colete": True, "duvidosa": cf < CONF_CERTA or confundida})
    return {"w": w, "h": h, "caixas": caixas + outros}


def cmd_rotular(args):
    todas = fotos_extraidas()
    if not todas:
        sys.exit("Nenhuma foto extraída. Rode antes:  python coleta.py extrair")
    revisao = ler_revisao()
    if args.refazer:  # refaz só o que ainda não foi revisado
        pendentes = [vf for vf in todas if vf not in revisao]
    else:
        pendentes = [vf for vf in todas if not (SUGERIDOS / vf[0] / f"{vf[1]}.json").exists()]
    if not pendentes:
        print("Todas as fotos já têm sugestão. Para refazer as não revisadas:  python coleta.py rotular --refazer")
        return

    perfil = escolher_perfil(args.gpu)
    from ultralytics import YOLO
    modelo = YOLO(args.modelo)
    cls_colete = [i for i, n in modelo.names.items() if categoria_da_classe(n) == "colete"]
    cls_confundidas = [i for i, n in modelo.names.items() if n in CONFUNDIDAS_COM_COLETE]
    if not cls_colete:
        sys.exit(f"O modelo {args.modelo} não tem classe de colete.")
    print(f"\nSugerindo caixas em {len(pendentes)} fotos com {args.modelo} · "
          f"{'GPU' if perfil['dispositivo'] != 'cpu' else 'CPU'} · imagem {perfil['imgsz']}")
    inicio, duvidosas = time.time(), 0
    for i, (video, foto) in enumerate(pendentes, 1):
        frame = cv2.imread(str(FOTOS / video / f"{foto}.jpg"))
        s = sugerir(modelo, frame, cenario_do_video(video), cls_colete, cls_confundidas,
                    perfil["dispositivo"], perfil["imgsz"])
        s["modelo"] = Path(args.modelo).name
        duvidosas += precisa_atencao(s, video)
        destino = SUGERIDOS / video / f"{foto}.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(s), encoding="utf-8")
        barra(i, len(pendentes), inicio, video[:28])
    print(f"\n  {len(pendentes) - duvidosas} com sugestão confiante · {duvidosas} para conferir com atenção")
    print("Próximo passo:  python coleta.py revisar   (ou  revisar --so-duvidosas  para começar pelas difíceis)")


def precisa_atencao(sugestao, video):
    """Foto com caixa duvidosa, ou sem colete achado num cenário que tem colete."""
    coletes = [c for c in sugestao["caixas"] if c["colete"]]
    if cenario_do_video(video) == NEGATIVO:
        return False
    return not coletes or any(c["duvidosa"] for c in coletes)


# --------------------------------------------------------------------------- #
# 4. revisar
# --------------------------------------------------------------------------- #
TECLAS_VOLTAR = {ord("b"), 2424832, 65361}      # b, seta esquerda (Windows, Linux)
TECLAS_PULAR = {ord("n"), 2555904, 65363}       # n, seta direita
TECLAS_ACEITAR = {13, 10, ord(" ")}


def desenhar_revisao(frame, caixas, escala, nomes, cabecalho, rodape):
    tela = cv2.resize(frame, None, fx=escala, fy=escala) if escala != 1 else frame.copy()
    for c in caixas:
        x1, y1, x2, y2 = (int(v * escala) for v in c["xyxy"])
        if c.get("colete"):
            cor = AMARELO if c.get("duvidosa") else VERDE
            cv2.rectangle(tela, (x1, y1), (x2, y2), cor, 3)
            rotulo = "colete" + (f" {c['conf']:.2f}" if "conf" in c else "") + (f" ({c['fonte']})"
                                                                                  if "fonte" in c else "")
            texto_com_fundo(tela, rotulo, (x1, y1), cor, escala=0.5)
        else:
            cv2.rectangle(tela, (x1, y1), (x2, y2), CINZA, 1)
            texto_com_fundo(tela, nomes.get(c["cls"], str(c["cls"])), (x1, y1), CINZA, escala=0.4)
    texto_com_fundo(tela, cabecalho, (10, 28), AMARELO, escala=0.6)
    texto_com_fundo(tela, rodape, (10, tela.shape[0] - 12), (60, 60, 60), escala=0.5)
    return tela


def cmd_revisar(args):
    revisao = ler_revisao()
    todas = [vf for vf in fotos_extraidas() if (SUGERIDOS / vf[0] / f"{vf[1]}.json").exists()]
    if not todas:
        sys.exit("Nenhuma foto com sugestão. Rode antes:  python coleta.py rotular")
    if args.cenario:
        todas = [vf for vf in todas if cenario_do_video(vf[0]) == args.cenario]
    if args.so_duvidosas:
        todas = [vf for vf in todas if precisa_atencao(ler_sugestao(*vf), vf[0])]
    if not args.tudo:
        todas = [vf for vf in todas if vf not in revisao]
    if not todas:
        print("Nada para revisar com esse filtro. Use --tudo para rever as já revisadas.")
        return

    nomes = nomes_das_classes()
    cls_colete = next(i for i, n in nomes.items() if categoria_da_classe(n) == "colete")
    janela = "Revisar colete padrao"
    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)
    rodape = "ENTER/espaco=aceitar  r=redesenhar coletes  x=sem colete  d=descartar  b/<-=voltar  n/->=pular  q=sair"
    i, feitas = 0, 0
    print(f"\n{len(todas)} fotos para revisar. Caixa verde = sugestão confiante, amarela = confira com atenção.")
    print("Caixa boa = justa no colete, do ombro até a barra, sem pegar braço nem fundo.\n")
    while 0 <= i < len(todas):
        video, foto = todas[i]
        frame = cv2.imread(str(FOTOS / video / f"{foto}.jpg"))
        h, w = frame.shape[:2]
        escala = min(1.0, 1280 / w, 720 / h)
        rev = REVISADOS / video / f"{foto}.txt"
        if rev.exists():  # já revisada: mostra o rótulo final
            caixas = [dict(c, colete=categoria_da_classe(nomes.get(c["cls"], "")) == "colete")
                      for c in ler_yolo(rev, w, h)]
        else:
            caixas = ler_sugestao(video, foto)["caixas"]
        corrigida = False
        anterior = revisao.get((video, foto), {}).get("status", "nova")
        while True:
            cab = f"{i + 1}/{len(todas)}  {cenario_do_video(video)}  [{anterior}]"
            if cenario_do_video(video) == NEGATIVO:
                cab += "  SEM COLETE: aceite sem caixa de colete"
            cv2.imshow(janela, desenhar_revisao(frame, caixas, escala, nomes, cab, rodape))
            tecla = cv2.waitKeyEx(0)
            if tecla in (ord("q"), 27):
                i = -1
                break
            if tecla in TECLAS_VOLTAR:
                i = max(0, i - 1)
                break
            if tecla in TECLAS_PULAR:
                i += 1
                break
            if tecla == ord("x"):
                caixas = [c for c in caixas if not c.get("colete")]
                corrigida = True
            if tecla == ord("r"):
                limpa = desenhar_revisao(frame, [c for c in caixas if not c.get("colete")], escala, nomes,
                                         "arraste sobre cada colete, ENTER confirma cada caixa, ESC termina",
                                         "")
                rois = cv2.selectROIs(janela, limpa, showCrosshair=False)
                if len(rois):
                    caixas = [c for c in caixas if not c.get("colete")] + [
                        {"cls": cls_colete, "xyxy": [x / escala, y / escala, (x + bw) / escala, (y + bh) / escala],
                         "colete": True, "duvidosa": False, "fonte": "manual"} for x, y, bw, bh in rois]
                    corrigida = True
            if tecla == ord("d") or tecla in TECLAS_ACEITAR:
                aceito = tecla != ord("d")
                if aceito:
                    if cenario_do_video(video) == NEGATIVO:
                        caixas = [c for c in caixas if not c.get("colete")]
                    rev.parent.mkdir(parents=True, exist_ok=True)
                    rev.write_text("\n".join(linha_yolo(c["xyxy"], c["cls"], w, h) for c in caixas),
                                   encoding="utf-8")
                elif rev.exists():
                    rev.unlink()
                revisao[(video, foto)] = {"video": video, "foto": foto,
                                          "status": "aceito" if aceito else "descartado",
                                          "corrigida": int(corrigida), "data": f"{datetime.now():%Y-%m-%d %H:%M}"}
                gravar_revisao(revisao)
                feitas += 1
                i += 1
                break
    cv2.destroyAllWindows()
    print(f"{feitas} fotos revisadas nesta sessão.")
    print("Próximo passo:  python coleta.py montar   (ou continue com  python coleta.py revisar)")


# --------------------------------------------------------------------------- #
# 5. montar
# --------------------------------------------------------------------------- #
def dividir_videos(lista):
    """Por cenário: ~FRACAO_VAL dos vídeos vão para a validação (pelo menos 1 se houver 2+)."""
    val = set()
    por_cenario = {}
    for v in sorted(lista):
        por_cenario.setdefault(cenario_do_video(v), []).append(v)
    sorteio = random.Random(SEMENTE)
    for vs in por_cenario.values():
        if len(vs) < 2:
            continue
        sorteio.shuffle(vs)
        val.update(vs[:max(1, round(len(vs) * FRACAO_VAL))])
    return val


def cmd_montar(args):
    aceitas = [k for k, l in ler_revisao().items() if l["status"] == "aceito"
               and (REVISADOS / k[0] / f"{k[1]}.txt").exists()]
    if not aceitas:
        sys.exit("Nenhuma foto aceita ainda. Rode:  python coleta.py revisar")
    val = dividir_videos({v for v, _ in aceitas})

    for sub in ("images", "labels"):
        if (DADOS / sub).exists():
            shutil.rmtree(DADOS / sub)  # dados_colete/ é sempre gerado de novo a partir da revisão
        for split in ("train", "val"):
            (DADOS / sub / split).mkdir(parents=True)
    resumo = {}
    for video, foto in aceitas:
        split = "val" if video in val else "train"
        nome = f"{video}__{foto}"  # começa pelo cenário: sem_colete__... = exemplo negativo
        shutil.copy2(FOTOS / video / f"{foto}.jpg", DADOS / "images" / split / f"{nome}.jpg")
        shutil.copy2(REVISADOS / video / f"{foto}.txt", DADOS / "labels" / split / f"{nome}.txt")
        r = resumo.setdefault(cenario_do_video(video), {"train": set(), "val": set(), "ft": 0, "fv": 0})
        r[split].add(video)
        r["fv" if split == "val" else "ft"] += 1
    escrever_yaml_treino(nomes_das_classes())

    print(f"\n{'cenário':<12} {'vídeos treino':>13} {'vídeos val':>10} {'fotos treino':>12} {'fotos val':>9}")
    for c in sorted(resumo):
        r = resumo[c]
        print(f"{c:<12} {len(r['train']):>13} {len(r['val']):>10} {r['ft']:>12} {r['fv']:>9}")
    total_v = sum(r["fv"] for r in resumo.values())
    print(f"\n{len(aceitas)} fotos em {DADOS.resolve()} ({total_v} na validação). dados_treino.yaml atualizado.")
    sem_val = [c for c, r in resumo.items() if not r["val"]]
    if sem_val:
        print(f"[AVISO] sem vídeo na validação (grave 2+ vídeos desses cenários): {', '.join(sorted(sem_val))}")
    print("Treine com:")
    print("  python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50")


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def cmd_status(args):
    revisao = ler_revisao()
    linhas = {c: {"videos": 0, "fotos": 0, "sug": 0, "ok": 0, "desc": 0} for c, _ in CENARIOS}
    for v in videos():
        linhas.setdefault(cenario_do_video(v.stem), {"videos": 0, "fotos": 0, "sug": 0, "ok": 0, "desc": 0})
        linhas[cenario_do_video(v.stem)]["videos"] += 1
    for video, foto in fotos_extraidas():
        l = linhas.setdefault(cenario_do_video(video), {"videos": 0, "fotos": 0, "sug": 0, "ok": 0, "desc": 0})
        l["fotos"] += 1
        l["sug"] += (SUGERIDOS / video / f"{foto}.json").exists()
        st = revisao.get((video, foto), {}).get("status")
        l["ok"] += st == "aceito"
        l["desc"] += st == "descartado"
    print(f"\n{'cenário':<12} {'vídeos':>6} {'fotos':>6} {'sugeridas':>9} {'aceitas':>7} {'descart.':>8} {'faltam':>6}")
    for c, l in linhas.items():
        print(f"{c:<12} {l['videos']:>6} {l['fotos']:>6} {l['sug']:>9} {l['ok']:>7} {l['desc']:>8} "
              f"{l['fotos'] - l['ok'] - l['desc']:>6}")
    for split in ("train", "val"):
        p = DADOS / "images" / split
        print(f"dados_colete/{split}: {len(list(p.glob('*.jpg'))) if p.is_dir() else 0} fotos")


# --------------------------------------------------------------------------- #
def main():
    comum = argparse.ArgumentParser(add_help=False)
    comum.add_argument("--gpu", action="store_true", help="usa a NVIDIA (RTX série 40 ou 50) no 'rotular'. Padrão: CPU")
    ap = argparse.ArgumentParser(description="Coleta do colete padrão em etapas")
    sub = ap.add_subparsers(dest="etapa", required=True)
    p = sub.add_parser("gravar", parents=[comum], help="grava vídeos por cenário")
    p.add_argument("--fonte", default="0", help="0/1/2 = webcam, ou URL de câmera IP")
    sub.add_parser("extrair", parents=[comum], help="tira fotos dos vídeos novos")
    p = sub.add_parser("rotular", parents=[comum], help="o modelo sugere as caixas")
    p.add_argument("--modelo", default=MODELO)
    p.add_argument("--refazer", action="store_true", help="refaz as sugestões das fotos ainda não revisadas")
    p = sub.add_parser("revisar", parents=[comum], help="confere as fotos uma a uma")
    p.add_argument("--cenario", choices=[c for c, _ in CENARIOS])
    p.add_argument("--so-duvidosas", action="store_true", help="só fotos com caixa amarela ou sem colete achado")
    p.add_argument("--tudo", action="store_true", help="inclui as já revisadas")
    sub.add_parser("montar", parents=[comum], help="gera dados_colete/ com as fotos aceitas")
    sub.add_parser("status", parents=[comum], help="resumo das etapas")
    args = ap.parse_args()
    {"gravar": cmd_gravar, "extrair": cmd_extrair, "rotular": cmd_rotular, "revisar": cmd_revisar,
     "montar": cmd_montar, "status": cmd_status}[args.etapa](args)


if __name__ == "__main__":
    main()
