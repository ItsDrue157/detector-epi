"""
Detector de EPI (capacete e colete) em tempo real com YOLO.

Exemplos de uso:
    python detectar_epi.py                          # webcam padrão (0)
    python detectar_epi.py --fonte 1                # segunda webcam
    python detectar_epi.py --fonte video.mp4        # arquivo de vídeo
    python detectar_epi.py --fonte foto.jpg         # imagem
    python detectar_epi.py --fonte rtsp://usuario:senha@ip:554/stream   # câmera IP
    python detectar_epi.py --exigir capacete        # cobra só capacete
    python detectar_epi.py --salvar saida.mp4       # grava o vídeo anotado
    python detectar_epi.py --dispositivo amd        # GPU AMD (Windows ML / ONNX)
    python detectar_epi.py --dispositivo cpu        # só processador

Teclas na janela:
    q ou ESC  -> sair
    p         -> pausar / continuar
    s         -> salvar um print da tela em ocorrencias/
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Cores (BGR)
# --------------------------------------------------------------------------- #
VERDE = (60, 200, 60)
VERMELHO = (40, 40, 230)
AMARELO = (0, 200, 255)
AZUL = (230, 160, 40)
CINZA = (160, 160, 160)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)

EXT_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Mapeamento das classes do modelo para categorias
# --------------------------------------------------------------------------- #
# Datasets de EPI usam nomes diferentes ("helmet", "Hardhat", "hard_hat",
# "NO-Hardhat", "no_helmet", "helmet_off", "Safety Vest", "no_vest"...).
# Em vez de fixar IDs, as classes são reconhecidas pelo nome, então o script
# funciona com o modelo treinado aqui ou com outros modelos de EPI.
PALAVRAS_CAPACETE = ("helmet", "hardhat", "hard_hat", "capacete")
PALAVRAS_COLETE = ("vest", "jacket", "colete")
PALAVRAS_PESSOA = ("person", "worker", "pessoa", "trabalhador")


def categoria_da_classe(nome: str) -> str | None:
    """Converte o nome de uma classe em: capacete, sem_capacete, colete,
    sem_colete, pessoa ou None (classe ignorada)."""
    n = re.sub(r"[^a-z]+", "_", nome.lower()).strip("_")
    negativo = bool(re.match(r"^(no|sem|without|not|missing)_", n)) or n.endswith("_off")
    if n in ("nohelmet", "nohardhat", "novest"):
        negativo = True

    if any(p in n for p in PALAVRAS_PESSOA):
        return "pessoa"
    if any(p in n for p in PALAVRAS_CAPACETE):
        return "sem_capacete" if negativo else "capacete"
    if any(p in n for p in PALAVRAS_COLETE):
        return "sem_colete" if negativo else "colete"
    return None


# --------------------------------------------------------------------------- #
# Estruturas
# --------------------------------------------------------------------------- #
@dataclass
class Deteccao:
    caixa: np.ndarray  # x1, y1, x2, y2
    conf: float
    categoria: str
    nome: str

    @property
    def centro(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.caixa
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class StatusPessoa:
    caixa: np.ndarray
    conf: float
    capacete: bool = False
    colete: bool = False
    viu_sem_capacete: bool = False
    viu_sem_colete: bool = False
    itens: list[Deteccao] = field(default_factory=list)

    def faltando(self, exigir: set[str]) -> list[str]:
        falta = []
        if "capacete" in exigir and (not self.capacete or self.viu_sem_capacete):
            falta.append("capacete")
        if "colete" in exigir and (not self.colete or self.viu_sem_colete):
            falta.append("colete")
        return falta


# --------------------------------------------------------------------------- #
# Lógica de associação EPI -> pessoa
# --------------------------------------------------------------------------- #
def _dentro(ponto, regiao) -> bool:
    x, y = ponto
    x1, y1, x2, y2 = regiao
    return x1 <= x <= x2 and y1 <= y <= y2


def avaliar_pessoas(deteccoes: list[Deteccao]) -> tuple[list[StatusPessoa], list[Deteccao]]:
    """Para cada pessoa detectada, verifica se há capacete na região da cabeça
    e colete na região do tronco. Retorna os status e as detecções de
    "sem EPI" que não ficaram dentro de nenhuma pessoa (órfãs)."""
    pessoas = [d for d in deteccoes if d.categoria == "pessoa"]
    epis = [d for d in deteccoes if d.categoria != "pessoa"]

    status = []
    usados: set[int] = set()
    for p in pessoas:
        x1, y1, x2, y2 = p.caixa
        w, h = x2 - x1, y2 - y1
        mx = 0.15 * w
        cabeca = (x1 - mx, y1 - 0.15 * h, x2 + mx, y1 + 0.40 * h)
        tronco = (x1 - mx, y1 + 0.10 * h, x2 + mx, y1 + 0.85 * h)

        s = StatusPessoa(caixa=p.caixa, conf=p.conf)
        melhor = {"capacete": 0.0, "sem_capacete": 0.0, "colete": 0.0, "sem_colete": 0.0}
        for i, e in enumerate(epis):
            regiao = cabeca if e.categoria in ("capacete", "sem_capacete") else tronco
            if _dentro(e.centro, regiao):
                melhor[e.categoria] = max(melhor[e.categoria], e.conf)
                s.itens.append(e)
                usados.add(i)

        # Se o modelo achou "com" e "sem" para a mesma pessoa, vence o mais confiante.
        s.capacete = melhor["capacete"] > 0 and melhor["capacete"] >= melhor["sem_capacete"]
        s.viu_sem_capacete = melhor["sem_capacete"] > melhor["capacete"]
        s.colete = melhor["colete"] > 0 and melhor["colete"] >= melhor["sem_colete"]
        s.viu_sem_colete = melhor["sem_colete"] > melhor["colete"]
        status.append(s)

    orfas = [
        e for i, e in enumerate(epis)
        if i not in usados and e.categoria in ("sem_capacete", "sem_colete")
    ]
    return status, orfas


# --------------------------------------------------------------------------- #
# Desenho
# --------------------------------------------------------------------------- #
def texto_com_fundo(img, texto, org, cor_fundo, escala=0.55, cor_texto=BRANCO, esp=1):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(texto, fonte, escala, esp)
    x, y = int(org[0]), int(org[1])
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y + 2), cor_fundo, -1)
    cv2.putText(img, texto, (x + 3, y - base), fonte, escala, cor_texto, esp, cv2.LINE_AA)
    return th + base + 6


def desenhar(frame, status, orfas, outras, exigir, mostrar_itens=True):
    for d in outras if mostrar_itens else []:
        cor = {"capacete": AZUL, "colete": AMARELO}.get(d.categoria, CINZA)
        x1, y1, x2, y2 = d.caixa.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 1)
        texto_com_fundo(frame, f"{d.nome} {d.conf:.2f}", (x1, y1), cor, escala=0.4)

    for s in status:
        falta = s.faltando(exigir)
        cor = VERDE if not falta else VERMELHO
        x1, y1, x2, y2 = s.caixa.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        linhas = []
        if "capacete" in exigir:
            linhas.append("CAPACETE OK" if "capacete" not in falta else "SEM CAPACETE")
        if "colete" in exigir:
            linhas.append("COLETE OK" if "colete" not in falta else "SEM COLETE")
        # rótulos acima da caixa (para não cobrir o rosto/capacete); dentro se não couber
        altura_rotulos = 22 * len(linhas)
        y = y1 - altura_rotulos - 2 if y1 - altura_rotulos - 2 > 36 else y1
        for ln in linhas:
            c = VERDE if "OK" in ln else VERMELHO
            y += texto_com_fundo(frame, ln, (x1, y + 18), c, escala=0.5)

    for d in orfas:
        x1, y1, x2, y2 = d.caixa.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), VERMELHO, 2)
        rotulo = "SEM CAPACETE" if d.categoria == "sem_capacete" else "SEM COLETE"
        texto_com_fundo(frame, f"{rotulo} {d.conf:.2f}", (x1, y1), VERMELHO, escala=0.5)


def desenhar_painel(frame, fps, n_pessoas, n_infratores, exigir, alerta_ativo, pausado):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), PRETO, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    info = (f"FPS: {fps:4.1f}   Pessoas: {n_pessoas}   Sem EPI: {n_infratores}   "
            f"Exigindo: {' + '.join(sorted(exigir))}")
    cv2.putText(frame, info, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BRANCO, 1, cv2.LINE_AA)
    if pausado:
        texto_com_fundo(frame, "PAUSADO", (w - 110, 26), AMARELO, escala=0.6, cor_texto=PRETO)
    if alerta_ativo:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), VERMELHO, 6)
        texto_com_fundo(frame, "ALERTA: EPI AUSENTE", (10, h - 15), VERMELHO, escala=0.8, esp=2)


# --------------------------------------------------------------------------- #
# Alertas e registro
# --------------------------------------------------------------------------- #
def bipe():
    def _tocar():
        try:
            if os.name == "nt":
                import winsound
                winsound.Beep(1200, 350)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass
    threading.Thread(target=_tocar, daemon=True).start()


class RegistroOcorrencias:
    def __init__(self, pasta: Path, intervalo_s: float, som: bool):
        self.pasta = pasta
        self.intervalo = intervalo_s
        self.som = som
        self.ultimo = 0.0
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.csv = self.pasta / "registro.csv"
        if not self.csv.exists():
            with open(self.csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f, delimiter=";").writerow(
                    ["data_hora", "pessoas", "sem_capacete", "sem_colete", "imagem"])

    def registrar(self, frame, n_pessoas, sem_cap, sem_col, forcar=False) -> bool:
        agora = time.time()
        if not forcar and agora - self.ultimo < self.intervalo:
            return False
        self.ultimo = agora
        carimbo = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        arq = self.pasta / f"ocorrencia_{carimbo}.jpg"
        cv2.imwrite(str(arq), frame)
        with open(self.csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter=";").writerow(
                [datetime.now().isoformat(timespec="seconds"), n_pessoas, sem_cap, sem_col, arq.name])
        if self.som and not forcar:
            bipe()
        print(f"[ALERTA] {carimbo}  sem capacete: {sem_cap}  sem colete: {sem_col}  -> {arq}")
        return True


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def preparar_onnx(caminho_modelo, imgsz):
    """Converte um .pt para ONNX (uma vez só) e devolve o caminho do .onnx.

    O ONNX roda na GPU AMD pelo Windows ML (pacote onnxruntime-windowsml), sem precisar de CUDA.
    """
    if not caminho_modelo or Path(caminho_modelo).suffix.lower() != ".pt" or not Path(caminho_modelo).exists():
        return caminho_modelo
    pt = Path(caminho_modelo)
    destino = pt.with_name(f"{pt.stem}_{imgsz}.onnx")
    if not destino.exists():
        from ultralytics import YOLO
        print(f"Convertendo {pt} para ONNX (só na primeira vez)...")
        gerado = Path(YOLO(str(pt)).export(format="onnx", imgsz=imgsz, device="cpu"))
        gerado.replace(destino)
    return str(destino)


def ativar_directml():
    """No Windows 10 (sem Windows ML) faz o ONNX usar a GPU pelo DirectML (pacote onnxruntime-directml).

    O Ultralytics só pede a CPU nesse caso, então trocamos a lista de providers antes de criar a sessão.
    """
    if sys.platform != "win32" or sys.getwindowsversion().build >= 26100:
        return  # Windows 11 24H2+ usa o Windows ML
    try:
        import onnxruntime as ort
    except ImportError:
        return
    if "DmlExecutionProvider" not in ort.get_available_providers():
        print("Aviso: DirectML não encontrado. Instale com:  pip install -r requirements-amd-win10.txt")
        return
    sessao_original = ort.InferenceSession

    def sessao_dml(modelo, sess_options=None, providers=None, *a, **kw):
        if not providers or providers == ["CPUExecutionProvider"]:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            sess_options = sess_options or ort.SessionOptions()
            sess_options.enable_mem_pattern = False  # exigências do DirectML
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return sessao_original(modelo, sess_options, providers, *a, **kw)

    ort.InferenceSession = sessao_dml
    print("GPU via DirectML (Windows 10).")


class DetectorEPI:
    def __init__(self, caminho_modelo, conf, iou, imgsz, dispositivo, modelo_pessoa=None, conf_categoria=None):
        from ultralytics import YOLO  # import aqui para o --help funcionar sem ultralytics

        self.modelo = YOLO(caminho_modelo)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        # confiança mínima própria por categoria (ex.: capacete mais rígido para não aceitar boné)
        self.conf_categoria = {k: v for k, v in (conf_categoria or {}).items() if v is not None}
        self.dispositivo = dispositivo
        self.nomes = self.modelo.names
        self.mapa = {i: categoria_da_classe(n) for i, n in self.nomes.items()}
        self.tem_pessoa = "pessoa" in self.mapa.values()
        # Só as classes reconhecidas (capacete, colete, pessoa e "sem EPI") entram na inferência.
        # As demais (ex.: Balaclava, C_eletrica, Operacional) nem são detectadas nem desenhadas.
        usadas = [i for i, c in self.mapa.items() if c]
        self.classes_usadas = usadas if len(usadas) < len(self.mapa) else None

        print("\nClasses usadas:")
        for i, n in self.nomes.items():
            if self.mapa[i]:
                print(f"  {i:>2}: {n:<18} -> {self.mapa[i]}")

        # Se o modelo de EPI não detecta pessoas, usa um segundo modelo (ex.: COCO) só para isso.
        self.modelo_pessoa, self.classes_pessoa = None, []
        if not self.tem_pessoa and modelo_pessoa and Path(modelo_pessoa).exists():
            m = YOLO(modelo_pessoa)
            self.classes_pessoa = [i for i, n in m.names.items() if categoria_da_classe(n) == "pessoa"]
            if self.classes_pessoa:
                self.modelo_pessoa = m
                self.tem_pessoa = True
                print(f"\nPessoas detectadas pelo modelo auxiliar '{modelo_pessoa}'.")

        if not self.tem_pessoa:
            print("\n[AVISO] Nenhum modelo detecta 'pessoa'. Serão contadas apenas as"
                  " detecções explícitas de 'sem capacete'/'sem colete'.")
            if not any(c in ("sem_capacete", "sem_colete") for c in self.mapa.values()):
                print("[AVISO] Este modelo também não tem classes 'sem capacete'/'sem colete':"
                      " nenhuma infração será detectada. Use --modelo-pessoa.")
        print()

    def detectar(self, frame) -> tuple[list[Deteccao], list[Deteccao]]:
        r = self.modelo.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                device=self.dispositivo, classes=self.classes_usadas,
                                verbose=False)[0]
        uteis, outras = [], []
        caixas = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
        for c, cf, cl in zip(caixas, confs, classes):
            cat = self.mapa.get(cl)
            if cf < self.conf_categoria.get(cat, 0.0):
                continue
            d = Deteccao(caixa=c, conf=float(cf), categoria=cat or "outra", nome=str(self.nomes[cl]))
            (uteis if cat else outras).append(d)

        if self.modelo_pessoa is not None:
            rp = self.modelo_pessoa.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                            device=self.dispositivo, classes=self.classes_pessoa,
                                            verbose=False)[0]
            if rp.boxes is not None:
                for c, cf in zip(rp.boxes.xyxy.cpu().numpy(), rp.boxes.conf.cpu().numpy()):
                    uteis.append(Deteccao(caixa=c, conf=float(cf), categoria="pessoa", nome="person"))
        return uteis, outras


def processar_frame(detector, frame, exigir, mostrar_itens):
    uteis, outras = detector.detectar(frame)
    status, orfas = avaliar_pessoas(uteis)
    itens_positivos = [d for d in uteis if d.categoria in ("capacete", "colete")]
    desenhar(frame, status, orfas, itens_positivos + outras, exigir, mostrar_itens)

    infratores = [s for s in status if s.faltando(exigir)]
    sem_cap = sum("capacete" in s.faltando(exigir) for s in status)
    sem_col = sum("colete" in s.faltando(exigir) for s in status)
    sem_cap += sum(o.categoria == "sem_capacete" and "capacete" in exigir for o in orfas)
    sem_col += sum(o.categoria == "sem_colete" and "colete" in exigir for o in orfas)
    n_infr = len(infratores) + sum(
        (o.categoria == "sem_capacete" and "capacete" in exigir)
        or (o.categoria == "sem_colete" and "colete" in exigir) for o in orfas)
    return len(status), n_infr, sem_cap, sem_col


# --------------------------------------------------------------------------- #
# Fontes de vídeo
# --------------------------------------------------------------------------- #
def abrir_captura(fonte: str, largura: int, altura: int):
    if fonte.isdigit():
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(fonte), backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap = cv2.VideoCapture(int(fonte))
        if largura and altura:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    else:
        cap = cv2.VideoCapture(fonte)
    return cap


def rodar_imagem(detector, caminho, args, exigir):
    frame = cv2.imread(caminho)
    if frame is None:
        sys.exit(f"Não consegui abrir a imagem: {caminho}")
    t0 = time.time()
    n_p, n_i, sc, scol = processar_frame(detector, frame, exigir, not args.ocultar_itens)
    desenhar_painel(frame, 1 / max(time.time() - t0, 1e-6), n_p, n_i, exigir, n_i > 0, False)
    saida = args.salvar or str(Path(caminho).with_name(Path(caminho).stem + "_epi.jpg"))
    cv2.imwrite(saida, frame)
    print(f"Pessoas: {n_p} | sem capacete: {sc} | sem colete: {scol} | resultado salvo em {saida}")
    if not args.sem_janela:
        cv2.namedWindow("Deteccao de EPI", cv2.WINDOW_NORMAL)
        cv2.imshow("Deteccao de EPI", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def rodar_video(detector, args, exigir):
    cap = abrir_captura(args.fonte, args.largura, args.altura)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir a fonte '{args.fonte}'. Se for webcam, tente --fonte 1 "
                 "ou feche outros programas que estejam usando a câmera.")

    gravador = None
    if args.salvar:
        fps_fonte = cap.get(cv2.CAP_PROP_FPS) or 0
        fps_fonte = fps_fonte if 1 < fps_fonte < 121 else 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        gravador = cv2.VideoWriter(args.salvar, cv2.VideoWriter_fourcc(*"mp4v"), fps_fonte, (w, h))

    registro = RegistroOcorrencias(Path(args.pasta_ocorrencias), args.intervalo_alerta,
                                   som=not args.sem_som)
    janela = "Deteccao de EPI"
    if not args.sem_janela:
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    fps, pausado, frames_infracao = 0.0, False, 0
    ultimo_frame = None
    print("Rodando... (q/ESC sai, p pausa, s salva print)")

    while True:
        if not pausado:
            ok, frame = cap.read()
            if not ok:
                if args.fonte.isdigit():
                    print("Falha ao ler a webcam.")
                break
            t0 = time.time()
            n_p, n_i, sc, scol = processar_frame(detector, frame, exigir, not args.ocultar_itens)
            dt = time.time() - t0
            fps = (1 / dt) if fps == 0 else 0.9 * fps + 0.1 * (1 / max(dt, 1e-6))

            # Só alerta se a infração persistir alguns frames (evita falso alarme por 1 frame ruim)
            frames_infracao = frames_infracao + 1 if n_i > 0 else 0
            alerta = frames_infracao >= args.frames_confirmacao
            desenhar_painel(frame, fps, n_p, n_i, exigir, alerta, False)
            if alerta:
                registro.registrar(frame, n_p, sc, scol)
            if gravador is not None:
                gravador.write(frame)
            ultimo_frame = frame
        elif ultimo_frame is not None:
            frame = ultimo_frame.copy()
            desenhar_painel(frame, fps, 0, 0, exigir, False, True)

        if args.sem_janela:
            continue
        cv2.imshow(janela, frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("p"):
            pausado = not pausado
        if tecla == ord("s") and ultimo_frame is not None:
            registro.registrar(ultimo_frame, 0, 0, 0, forcar=True)
        if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    if gravador is not None:
        gravador.release()
        print(f"Vídeo salvo em {args.salvar}")
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Detecção de EPI (capacete e colete) com YOLO")
    ap.add_argument("--modelo", default="modelos/epi.pt", help="arquivo .pt do modelo de EPI")
    ap.add_argument("--fonte", default="0", help="0/1/2 = webcam, ou caminho de vídeo/imagem, ou URL rtsp")
    ap.add_argument("--modelo-pessoa", default="modelos/yolo26s.pt",
                    help="modelo auxiliar (ex.: COCO) para detectar pessoas, usado só se o modelo"
                         " de EPI não tiver classe 'pessoa'. 'nenhum' desativa")
    ap.add_argument("--conf", type=float, default=0.40, help="confiança mínima (0-1)")
    ap.add_argument("--conf-capacete", type=float, default=None,
                    help="confiança mínima só para capacete (ex.: 0.7 se boné/chapéu vira capacete)")
    ap.add_argument("--conf-colete", type=float, default=None,
                    help="confiança mínima só para colete (ex.: 0.6 se camisa colorida vira colete)")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU do NMS")
    ap.add_argument("--imgsz", type=int, default=640, help="tamanho de entrada do modelo")
    ap.add_argument("--dispositivo", default=None, help="'cpu', '0' (GPU NVIDIA 0 ou AMD com ROCm), 'amd' (GPU AMD via"
                         " Windows ML/ONNX), 'mps' (Mac). Padrão: automático")
    ap.add_argument("--exigir", nargs="+", default=["capacete", "colete"],
                    choices=["capacete", "colete"], help="quais EPIs são obrigatórios")
    ap.add_argument("--largura", type=int, default=1280, help="largura pedida à webcam")
    ap.add_argument("--altura", type=int, default=720, help="altura pedida à webcam")
    ap.add_argument("--salvar", default=None, help="salva o vídeo/imagem anotado neste caminho")
    ap.add_argument("--pasta-ocorrencias", default="ocorrencias", help="onde salvar fotos e o CSV de alertas")
    ap.add_argument("--intervalo-alerta", type=float, default=5.0, help="segundos mínimos entre alertas")
    ap.add_argument("--frames-confirmacao", type=int, default=8,
                    help="nº de frames seguidos com infração antes de alertar")
    ap.add_argument("--sem-som", action="store_true", help="desliga o bipe")
    ap.add_argument("--sem-janela", action="store_true", help="não abre janela (servidor)")
    ap.add_argument("--ocultar-itens", action="store_true", help="não desenha as caixas de capacete/colete")
    args = ap.parse_args()

    if not Path(args.modelo).exists() and not args.modelo.startswith(("http", "hf://")):
        sys.exit(f"Modelo '{args.modelo}' não encontrado.\n"
                 "Treine primeiro com:  python treinar.py\n"
                 "ou aponte para outro modelo com --modelo caminho/do/modelo.pt")

    modelo_pessoa = None if args.modelo_pessoa.lower() == "nenhum" else args.modelo_pessoa
    dispositivo = args.dispositivo
    if dispositivo and dispositivo.lower() in ("amd", "onnx", "winml"):
        # O Windows ML escolhe sozinho a GPU (AMD, Intel ou NVIDIA) e cai para a CPU se não houver.
        args.modelo = preparar_onnx(args.modelo, args.imgsz)
        modelo_pessoa = preparar_onnx(modelo_pessoa, args.imgsz)
        ativar_directml()
        dispositivo = "cpu"
    detector = DetectorEPI(args.modelo, args.conf, args.iou, args.imgsz, dispositivo, modelo_pessoa,
                           {"capacete": args.conf_capacete, "colete": args.conf_colete})
    exigir = set(args.exigir)

    if Path(args.fonte).suffix.lower() in EXT_IMAGEM:
        rodar_imagem(detector, args.fonte, args, exigir)
    else:
        rodar_video(detector, args, exigir)


if __name__ == "__main__":
    main()
