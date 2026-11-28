"""
Coleta fotos do NOSSO COLETE PADRÃO (verde-limão com faixas refletivas), para o
modelo aprender a achá-lo. Hoje o modelo original só o acha com confiança de 0.03 a 0.10.

Cada foto salva já vai rotulada com o que o modelo achou. Como aqui só aparece
o colete padrão, qualquer detecção de colete vale (a partir de 0.02).
Nas costas, o modelo às vezes confunde o colete com a jaqueta de eletricista
(C_eletrica) ou com o uniforme Operacional: aqui essas caixas viram colete.
Com o colete aberto, ele acha as duas metades separadas: aqui elas viram uma caixa só.
Se a caixa ficar errada ou ele não achar, aperte "m": a imagem congela e você
desenha a caixa com o mouse. Essas fotos são as que mais ensinam.

IMPORTANTE: só filme pessoas com o colete padrão. Deixe outros objetos
verde-limão fora da imagem.

Exemplos:
    python coletar_colete.py                    # webcam
    python coletar_colete.py --fonte obra.mp4   # vídeo gravado (1 foto por segundo)

Teclas:
    espaço = salvar foto | a = liga/desliga salvar automático | q/ESC = sair
    m      = marcar à mão: arraste uma caixa sobre cada colete, ENTER confirma cada caixa,
             ESC termina e salva (ESC sem nenhuma caixa cancela)

Depois:
    python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from detectar_epi import AMARELO, VERDE, VERMELHO, abrir_captura, categoria_da_classe, texto_com_fundo

PASTA = Path("dados_colete")
CONF_COLETE = 0.02  # só aparece o colete padrão, então até detecção bem fraca vale
CONF_OUTROS = 0.40  # confiança mínima dos rótulos dos outros EPIs (capacete etc.)
# Uniformes que o modelo confunde com o colete padrão (costas amarelas com refletivo).
# Na coleta só aparece o colete padrão, então essas caixas são o colete.
CONFUNDIDAS_COM_COLETE = ("C_eletrica", "Operacional")
CONF_CONFUNDIDAS = 0.25  # abaixo disso elas só alargariam a caixa do colete com lixo


def escrever_yaml_treino(nomes):
    """dados_treino.yaml: dataset original + fotos de boné + fotos de colete."""
    extras = [p for p in ("dados_bone", "dados_colete") if Path(p, "images").is_dir()]
    linhas = ["# dataset de treino: original + fotos coletadas", "train:", "  - teste/train/images"]
    linhas += [f"  - {p}/images/train" for p in extras]
    linhas += ["val:", "  - teste/valid/images"]
    linhas += [f"  - {p}/images/val" for p in extras]
    linhas += [f"nc: {len(nomes)}", f"names: {[nomes[i] for i in sorted(nomes)]}"]
    Path("dados_treino.yaml").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def linha_yolo(caixa, cl, w, h):
    x1, y1, x2, y2 = caixa
    return f"{cl} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def juntar_caixas(caixas):
    """Junta caixas de colete que se sobrepõem ou ficam lado a lado (colete aberto:
    o modelo acha cada metade separada). Recebe e devolve [(x1, y1, x2, y2, conf)]."""
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
                    caixas[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]),
                                 max(a[4], b[4])]
                    del caixas[j]
                    juntou = True
                    break
            if juntou:
                break
    return caixas


def main():
    ap = argparse.ArgumentParser(description="Coleta fotos do colete padrão, já rotuladas")
    ap.add_argument("--fonte", default="0", help="0/1/2 = webcam, ou caminho de vídeo")
    args = ap.parse_args()

    if not Path("teste/train/images").exists():
        sys.exit("Não achei teste/train/images (dataset original).")

    from ultralytics import YOLO
    modelo = YOLO("modelos/epi.pt")
    cls_colete = [i for i, n in modelo.names.items() if categoria_da_classe(n) == "colete"]
    cls_confundidas = [i for i, n in modelo.names.items() if n in CONFUNDIDAS_COM_COLETE]

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (PASTA / sub).mkdir(parents=True, exist_ok=True)
    escrever_yaml_treino(modelo.names)
    salvas = len(list((PASTA / "images").rglob("*.jpg")))

    eh_video = not args.fonte.isdigit()
    cap = abrir_captura(args.fonte)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir a fonte '{args.fonte}'.")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    auto = eh_video  # em vídeo já começa salvando sozinho
    ultimo, n_frame = 0.0, 0
    janela = "Coletar colete padrao"

    def salvar(frame, rotulos):
        nonlocal salvas
        split = "val" if salvas % 5 == 4 else "train"  # 1 em cada 5 vai para validação
        nome = f"colete_{datetime.now():%Y%m%d_%H%M%S_%f}"
        cv2.imwrite(str(PASTA / "images" / split / f"{nome}.jpg"), frame)
        (PASTA / "labels" / split / f"{nome}.txt").write_text("\n".join(rotulos), encoding="utf-8")
        salvas += 1

    print(f"\nSalvando em {PASTA.resolve()}  (já existem {salvas} fotos)")
    print("Só filme pessoas com o COLETE PADRÃO (verde-limão com faixas refletivas).\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frame += 1
        h, w = frame.shape[:2]
        r = modelo.predict(frame, conf=CONF_COLETE, verbose=False)[0]
        rotulos_outros, rotulos, coletes, tela = [], [], [], frame.copy()
        for caixa, cl, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                                 r.boxes.conf.cpu().numpy()):
            p1, p2 = tuple(caixa[:2].astype(int)), tuple(caixa[2:].astype(int))
            if cl in cls_colete or (cl in cls_confundidas and cf >= CONF_CONFUNDIDAS):
                coletes.append((*caixa, cf))  # C_eletrica/Operacional aqui = o colete padrão
            elif cf >= CONF_OUTROS and cl not in cls_confundidas:
                rotulos_outros.append(linha_yolo(caixa, cl, w, h))
                cv2.rectangle(tela, p1, p2, VERDE, 1)
                texto_com_fundo(tela, f"{modelo.names[cl]} (rotulo)", p1, VERDE, escala=0.4)
        for x1, y1, x2, y2, cf in juntar_caixas(coletes):
            rotulos.append(linha_yolo((x1, y1, x2, y2), cls_colete[0], w, h))
            p1 = (int(x1), int(y1))
            cv2.rectangle(tela, p1, (int(x2), int(y2)), VERDE, 3)
            texto_com_fundo(tela, f"colete padrao {cf:.2f}", p1, VERDE, escala=0.5)
        achou = len(rotulos) > 0
        rotulos += rotulos_outros
        if achou:
            texto_com_fundo(tela, "caixa errada ou folgada? aperte M para marcar a mao",
                            (10, h - 15), VERDE, escala=0.6)
        else:
            texto_com_fundo(tela, "colete nao encontrado: aperte M para marcar a mao",
                            (10, h - 15), VERMELHO, escala=0.6)

        agora = n_frame / fps_video if eh_video else time.time()
        quer_salvar = auto and achou and agora - ultimo >= 1.0  # no automático só salva se achou o colete
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("a"):
            auto = not auto
        if tecla == ord(" "):
            quer_salvar = True
        if tecla == ord("m"):
            marcar = frame.copy()
            texto_com_fundo(marcar, "arraste sobre cada colete, ENTER confirma, ESC termina",
                            (10, 30), AMARELO, escala=0.6)
            caixas = cv2.selectROIs(janela, marcar, showCrosshair=False)
            if len(caixas):
                manuais = [linha_yolo((x, y, x + bw, y + bh), cls_colete[0], w, h) for x, y, bw, bh in caixas]
                salvar(frame, manuais + rotulos_outros)
                print(f"  {len(caixas)} colete(s) marcado(s) à mão")
            quer_salvar = False

        if quer_salvar:
            ultimo = agora
            salvar(frame, rotulos)
            cv2.rectangle(tela, (0, 0), (w - 1, h - 1), VERDE, 8)

        texto_com_fundo(tela, f"fotos: {salvas} | automatico: {'SIM' if auto else 'nao'} | "
                              "espaco=salvar a=auto m=marcar q=sair", (10, 30), AMARELO, escala=0.6)
        cv2.imshow(janela, tela)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{salvas} fotos em {PASTA.resolve()}")
    print("Agora treine com:")
    print("  python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50")


if __name__ == "__main__":
    main()
