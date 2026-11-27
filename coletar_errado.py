"""
Coleta fotos do que PARECE EPI mas NÃO É, para o modelo parar de confundir:
    --item colete    -> colete de futebol (padrão)  -> dados_colete_errado/
    --item capacete  -> boné                        -> dados_bone/

Tudo que o modelo achar como o item escolhido é tratado como ERRADO e fica sem
rótulo (riscado em vermelho na tela). Os outros EPIs que ele achar viram rótulo.

IMPORTANTE: não deixe aparecer o EPI de verdade desse item (colete de EPI com
--item colete, capacete com --item capacete). Ele seria ensinado como "não é EPI".

Exemplos:
    python coletar_errado.py                             # colete de futebol, webcam
    python coletar_errado.py --fonte futebol.mp4         # vídeo gravado (1 foto por segundo)
    python coletar_errado.py --item capacete             # bonés

Teclas: espaço = salvar foto | a = liga/desliga salvar automático | q/ESC = sair

Depois:
    python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50
"""

import argparse

import cv2

from coleta import PASTAS, classes_do_item, coletar, linha_yolo
from detectar_epi import VERDE, VERMELHO, texto_com_fundo


def main():
    ap = argparse.ArgumentParser(description="Coleta fotos do que parece EPI mas não é (rótulo de errado)")
    ap.add_argument("--item", default="colete", choices=["colete", "capacete"],
                    help="colete (colete de futebol) ou capacete (boné)")
    ap.add_argument("--fonte", default="0", help="0/1/2 = webcam, ou caminho de vídeo")
    ap.add_argument("--conf", type=float, default=0.40, help="confiança mínima dos rótulos dos outros EPIs")
    args = ap.parse_args()
    falsas = None

    def rotular(frame, modelo):
        nonlocal falsas
        if falsas is None:
            falsas = set(classes_do_item(modelo, args.item))
        r = modelo.predict(frame, conf=args.conf, verbose=False)[0]
        h, w = frame.shape[:2]
        rotulos, tela = [], frame.copy()
        for caixa, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            p1, p2 = tuple(caixa[:2].astype(int)), tuple(caixa[2:].astype(int))
            if cl in falsas:  # colete de futebol / boné que o modelo achou que era EPI: fica sem rótulo
                cv2.rectangle(tela, p1, p2, VERMELHO, 2)
                cv2.line(tela, p1, p2, VERMELHO, 2)
                texto_com_fundo(tela, f"{args.item} -> ERRADO", p1, VERMELHO, escala=0.45)
                continue
            cv2.rectangle(tela, p1, p2, VERDE, 2)
            texto_com_fundo(tela, f"{modelo.names[cl]} (rotulo)", p1, VERDE, escala=0.45)
            rotulos.append(linha_yolo(caixa, cl, w, h))
        return rotulos, tela, True

    o_que = PASTAS[(args.item, "errado")]["o_que"]
    print(f"Só filme pessoas de {o_que.upper()}, sem {args.item} de EPI de verdade.")
    coletar(args.item, "errado", args.fonte, rotular)


if __name__ == "__main__":
    main()
