"""
Coleta fotos de EPI DE VERDADE, para o modelo achar o EPI que hoje ele perde:
    --item colete    -> colete de EPI (padrão)  -> dados_colete_certo/
    --item capacete  -> capacete de EPI         -> dados_capacete_certo/

Como aqui só aparece EPI de verdade, o modelo aceita até detecções fracas do item
(confiança a partir de 0.15). Se mesmo assim ele não achar, aperte "m": a imagem
congela e você desenha a caixa com o mouse. Essas fotos são as que mais ensinam.

IMPORTANTE: só filme o EPI de verdade (nada de colete de futebol / boné aqui).

Exemplos:
    python coletar_certo.py                       # colete de EPI, webcam
    python coletar_certo.py --fonte obra.mp4      # vídeo gravado (1 foto por segundo)
    python coletar_certo.py --item capacete

Teclas:
    espaço = salvar foto | a = liga/desliga salvar automático | q/ESC = sair
    m      = marcar à mão: arraste uma caixa sobre cada EPI, ENTER confirma cada caixa,
             ESC termina e salva (ESC sem nenhuma caixa cancela)

Depois:
    python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50
"""

import argparse

import cv2

from coleta import PASTAS, classes_do_item, coletar, linha_yolo
from detectar_epi import AMARELO, VERDE, VERMELHO, texto_com_fundo

CONF_ITEM = 0.15  # aqui todo item na imagem é EPI de verdade, então detecção fraca também vale


def main():
    ap = argparse.ArgumentParser(description="Coleta fotos de EPI de verdade (rótulo de certo)")
    ap.add_argument("--item", default="colete", choices=["colete", "capacete"],
                    help="colete ou capacete de EPI")
    ap.add_argument("--fonte", default="0", help="0/1/2 = webcam, ou caminho de vídeo")
    ap.add_argument("--conf", type=float, default=0.40, help="confiança mínima dos rótulos dos outros EPIs")
    args = ap.parse_args()
    do_item = None

    def outros_rotulos(r, w, h):
        """Rótulos dos outros EPIs (não o item), com a confiança normal."""
        linhas, caixas = [], []
        for caixa, cl, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                                 r.boxes.conf.cpu().numpy()):
            if cl not in do_item and cf >= args.conf:
                linhas.append(linha_yolo(caixa, cl, w, h))
                caixas.append((caixa, cl))
        return linhas, caixas

    def rotular(frame, modelo):
        nonlocal do_item
        if do_item is None:
            do_item = classes_do_item(modelo, args.item)
        r = modelo.predict(frame, conf=min(CONF_ITEM, args.conf), verbose=False)[0]
        h, w = frame.shape[:2]
        rotulos, caixas = outros_rotulos(r, w, h)
        tela = frame.copy()
        for caixa, cl in caixas:
            cv2.rectangle(tela, tuple(caixa[:2].astype(int)), tuple(caixa[2:].astype(int)), VERDE, 1)
            texto_com_fundo(tela, f"{modelo.names[cl]} (rotulo)", caixa[:2], VERDE, escala=0.4)

        achou = 0
        for caixa, cl, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                                 r.boxes.conf.cpu().numpy()):
            if cl in do_item and cf >= CONF_ITEM:
                rotulos.append(linha_yolo(caixa, cl, w, h))
                achou += 1
                cv2.rectangle(tela, tuple(caixa[:2].astype(int)), tuple(caixa[2:].astype(int)), VERDE, 3)
                texto_com_fundo(tela, f"{args.item} CERTO {cf:.2f}", caixa[:2], VERDE, escala=0.5)
        if not achou:
            h_t = tela.shape[0]
            texto_com_fundo(tela, f"{args.item} nao encontrado: aperte M para marcar a mao",
                            (10, h_t - 15), VERMELHO, escala=0.6)
        return rotulos, tela, achou > 0  # no automático só salva quando o item foi achado

    def marcar_a_mao(tecla, frame, modelo, janela):
        if tecla != ord("m"):
            return None
        tela = frame.copy()
        texto_com_fundo(tela, f"arraste sobre cada {args.item}, ENTER confirma, ESC termina",
                        (10, 30), AMARELO, escala=0.6)
        caixas = cv2.selectROIs(janela, tela, showCrosshair=False)
        if len(caixas) == 0:
            return None
        h, w = frame.shape[:2]
        r = modelo.predict(frame, conf=args.conf, verbose=False)[0]
        rotulos, _ = outros_rotulos(r, w, h)
        rotulos += [linha_yolo((x, y, x + bw, y + bh), do_item[0], w, h) for x, y, bw, bh in caixas]
        print(f"  {len(caixas)} {args.item}(s) marcado(s) à mão")
        return rotulos

    o_que = PASTAS[(args.item, "certo")]["o_que"]
    print(f"Só filme pessoas de {o_que.upper()} (nada de colete de futebol / boné).")
    coletar(args.item, "certo", args.fonte, rotular, tecla_extra=marcar_a_mao)


if __name__ == "__main__":
    main()
