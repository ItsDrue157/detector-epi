"""
Detector de EPI (capacete e colete) em tempo real com YOLO.

Exemplos de uso:
    python detectar_epi.py                          # webcam padrão (0), na CPU
    python detectar_epi.py --gpu                    # placa NVIDIA RTX série 40 ou 50
    python detectar_epi.py --fonte 1                # segunda webcam
    python detectar_epi.py --fonte video.mp4        # arquivo de vídeo
    python detectar_epi.py --fonte foto.jpg         # imagem
    python detectar_epi.py --fonte rtsp://usuario:senha@ip:554/stream   # câmera IP
    python detectar_epi.py --exigir capacete        # cobra só capacete
    python detectar_epi.py --salvar saida.mp4       # grava o vídeo anotado
    python detectar_epi.py --sem-som                # sem bipe nos alertas

Ajustes finos (confiança, intervalo de alertas, resolução da câmera...) ficam
nas constantes da seção "Configuração", logo abaixo dos imports.

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
# Configuração
# --------------------------------------------------------------------------- #
MODELO_EPI = "modelos/epi.pt"
CONF = 0.40                 # confiança mínima geral (0-1)
CONF_CAPACETE = None        # confiança mínima só do capacete (ex.: 0.7 se boné virar capacete)
CONF_COLETE = None          # confiança mínima só do colete (ex.: 0.6 se camisa colorida virar colete)
IOU = 0.50                  # IoU do NMS
LARGURA, ALTURA = 1280, 720  # resolução pedida à webcam
PASTA_OCORRENCIAS = "ocorrencias"
INTERVALO_ALERTA = 5.0      # segundos mínimos entre dois alertas salvos
FRAMES_CONFIRMACAO = 8      # nº de frames seguidos com infração antes de alertar

# Perfis de hardware.
#   cpu: PC sem placa de vídeo (apresentação). Imagem 480 (mAP50 igual ao 640 no teste),
#        detector de pessoa nano rodando a cada 3 frames (pessoas mudam pouco entre frames).
#   gpu: NVIDIA RTX série 40 ou 50. Imagem 640, detector de pessoa "s" em todo frame.
PERFIS = {
    "cpu": {"dispositivo": "cpu", "imgsz": 480, "modelo_pessoa": "modelos/yolo26n.pt", "pessoa_a_cada": 3},
    "gpu": {"dispositivo": "0", "imgsz": 640, "modelo_pessoa": "modelos/yolo26s.pt", "pessoa_a_cada": 1},
}

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


def desenhar(frame, status, orfas, outras, exigir):
    for d in outras:
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
    faixa = frame[:34]  # escurece só a faixa do topo, sem copiar o frame inteiro
    cv2.addWeighted(faixa, 0.45, np.zeros_like(faixa), 0.55, 0, faixa)
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
class DetectorEPI:
    def __init__(self, caminho_modelo, conf, iou, imgsz, dispositivo, modelo_pessoa=None, conf_categoria=None,
                 pessoa_a_cada=1):
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
        # o detector de pessoa roda a cada N frames; nos outros reaproveita as últimas caixas
        self.pessoa_a_cada = max(1, pessoa_a_cada)
        self._n_frame, self._pessoas = 0, []
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
                      " nenhuma infração será detectada. Confira modelos/yolo26n.pt e yolo26s.pt.")
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
            if self._n_frame % self.pessoa_a_cada == 0:
                rp = self.modelo_pessoa.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                                device=self.dispositivo, classes=self.classes_pessoa,
                                                verbose=False)[0]
                self._pessoas = []
                if rp.boxes is not None:
                    for c, cf in zip(rp.boxes.xyxy.cpu().numpy(), rp.boxes.conf.cpu().numpy()):
                        self._pessoas.append(Deteccao(caixa=c, conf=float(cf), categoria="pessoa", nome="person"))
            self._n_frame += 1
            uteis.extend(self._pessoas)
        return uteis, outras


def processar_frame(detector, frame, exigir):
    uteis, outras = detector.detectar(frame)
    status, orfas = avaliar_pessoas(uteis)
    itens_positivos = [d for d in uteis if d.categoria in ("capacete", "colete")]
    desenhar(frame, status, orfas, itens_positivos + outras, exigir)

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
def abrir_captura(fonte: str, largura: int = LARGURA, altura: int = ALTURA):
    if fonte.isdigit():
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(fonte), backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap = cv2.VideoCapture(int(fonte))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    else:
        cap = cv2.VideoCapture(fonte)
    return cap


class CapturaAoVivo:
    """Lê a câmera numa thread e guarda só o frame mais recente.

    Sem isso, quando o processamento é mais lento que a câmera (CPU), os frames
    se acumulam no buffer e a imagem na tela fica atrasada em relação à realidade.
    """

    def __init__(self, cap):
        self.cap = cap
        self.ok, self.frame = False, None
        self.trava = threading.Lock()
        self.novo = threading.Event()
        self.rodando = True
        self.thread = threading.Thread(target=self._ler, daemon=True)
        self.thread.start()

    def _ler(self):
        while self.rodando:
            ok, frame = self.cap.read()
            with self.trava:
                self.ok, self.frame = ok, frame
            self.novo.set()
            if not ok:
                break

    def read(self):
        self.novo.wait(timeout=5)
        self.novo.clear()
        with self.trava:
            return self.ok, self.frame

    def release(self):
        self.rodando = False
        self.thread.join(timeout=1)
        self.cap.release()


def rodar_imagem(detector, caminho, args, exigir):
    frame = cv2.imread(caminho)
    if frame is None:
        sys.exit(f"Não consegui abrir a imagem: {caminho}")
    t0 = time.time()
    n_p, n_i, sc, scol = processar_frame(detector, frame, exigir)
    desenhar_painel(frame, 1 / max(time.time() - t0, 1e-6), n_p, n_i, exigir, n_i > 0, False)
    saida = args.salvar or str(Path(caminho).with_name(Path(caminho).stem + "_epi.jpg"))
    cv2.imwrite(saida, frame)
    print(f"Pessoas: {n_p} | sem capacete: {sc} | sem colete: {scol} | resultado salvo em {saida}")
    cv2.namedWindow("Deteccao de EPI", cv2.WINDOW_NORMAL)
    cv2.imshow("Deteccao de EPI", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rodar_video(detector, args, exigir):
    cap = abrir_captura(args.fonte)
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

    # webcam e câmera IP: sempre o frame mais recente. Arquivo de vídeo: todos os frames, em ordem.
    if args.fonte.isdigit() or args.fonte.lower().startswith(("rtsp://", "http://", "https://")):
        cap = CapturaAoVivo(cap)

    registro = RegistroOcorrencias(Path(PASTA_OCORRENCIAS), INTERVALO_ALERTA, som=not args.sem_som)
    janela = "Deteccao de EPI"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    fps, pausado, frames_infracao = 0.0, False, 0
    ultimo_frame, t_anterior = None, None
    print("Rodando... (q/ESC sai, p pausa, s salva print)")

    while True:
        if not pausado:
            ok, frame = cap.read()
            if not ok:
                if args.fonte.isdigit():
                    print("Falha ao ler a webcam.")
                break
            n_p, n_i, sc, scol = processar_frame(detector, frame, exigir)
            # FPS real do loop (leitura + detecção + janela), não só da detecção
            agora = time.time()
            if t_anterior is not None:
                dt = max(agora - t_anterior, 1e-6)
                fps = (1 / dt) if fps == 0 else 0.9 * fps + 0.1 * (1 / dt)
            t_anterior = agora

            # Só alerta se a infração persistir alguns frames (evita falso alarme por 1 frame ruim)
            frames_infracao = frames_infracao + 1 if n_i > 0 else 0
            alerta = frames_infracao >= FRAMES_CONFIRMACAO
            desenhar_painel(frame, fps, n_p, n_i, exigir, alerta, False)
            if alerta:
                registro.registrar(frame, n_p, sc, scol)
            if gravador is not None:
                gravador.write(frame)
            ultimo_frame = frame
        elif ultimo_frame is not None:
            frame = ultimo_frame.copy()
            desenhar_painel(frame, fps, 0, 0, exigir, False, True)

        cv2.imshow(janela, frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("p"):
            pausado = not pausado
            t_anterior = None  # o tempo pausado não entra no FPS
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
def escolher_perfil(gpu: bool) -> dict:
    """Perfil cpu por padrão; com --gpu, usa a NVIDIA se ela existir (feito para RTX série 40 ou 50)."""
    if not gpu:
        return PERFIS["cpu"]
    import torch
    if not torch.cuda.is_available():
        print("[AVISO] --gpu pedido, mas nenhuma placa NVIDIA (CUDA) foi encontrada. Rodando na CPU.\n"
              "        Confira se instalou com:  pip install -r requirements-gpu.txt")
        return PERFIS["cpu"]
    if torch.cuda.get_device_capability(0) < (8, 9):  # 8.9 = RTX série 40 (Ada)
        print(f"[AVISO] {torch.cuda.get_device_name(0)} é anterior à RTX série 40, que é o mínimo testado.")
    return PERFIS["gpu"]


def main():
    ap = argparse.ArgumentParser(description="Detecção de EPI (capacete e colete) com YOLO")
    ap.add_argument("--fonte", default="0",
                    help="0/1/2 = webcam, caminho de vídeo ou foto, ou URL da câmera IP (rtsp://...)")
    ap.add_argument("--gpu", action="store_true", help="usa a placa NVIDIA (RTX série 40 ou 50). Padrão: CPU")
    ap.add_argument("--exigir", nargs="+", default=["capacete", "colete"],
                    choices=["capacete", "colete"], help="quais EPIs são obrigatórios")
    ap.add_argument("--salvar", default=None, help="salva o vídeo/foto com as marcações neste caminho")
    ap.add_argument("--sem-som", action="store_true", help="desliga o bipe dos alertas")
    args = ap.parse_args()

    if not Path(MODELO_EPI).exists():
        sys.exit(f"Modelo '{MODELO_EPI}' não encontrado. Treine primeiro com:  python treinar.py")

    perfil = escolher_perfil(args.gpu)
    modelo_pessoa = perfil["modelo_pessoa"]
    reserva = PERFIS["gpu"]["modelo_pessoa"]
    if not Path(modelo_pessoa).exists() and Path(reserva).exists():
        print(f"[AVISO] '{modelo_pessoa}' não encontrado, usando '{reserva}' para pessoas (mais lento).")
        modelo_pessoa = reserva
    print(f"Rodando na {'GPU' if perfil['dispositivo'] != 'cpu' else 'CPU'} · imagem {perfil['imgsz']} · "
          f"pessoas: {modelo_pessoa} a cada {perfil['pessoa_a_cada']} frame(s)")

    detector = DetectorEPI(MODELO_EPI, CONF, IOU, perfil["imgsz"], perfil["dispositivo"], modelo_pessoa,
                           {"capacete": CONF_CAPACETE, "colete": CONF_COLETE}, perfil["pessoa_a_cada"])
    exigir = set(args.exigir)

    if Path(args.fonte).suffix.lower() in EXT_IMAGEM:
        rodar_imagem(detector, args.fonte, args, exigir)
    else:
        rodar_video(detector, args, exigir)


if __name__ == "__main__":
    main()
