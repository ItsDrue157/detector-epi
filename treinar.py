"""
Treina o modelo YOLO de detecção de EPI.

Por padrão usa o dataset aberto "Construction-PPE" da Ultralytics, que é
baixado automaticamente na primeira execução (capacete, colete, luvas,
botas, óculos, pessoa e as classes de "sem EPI").

Exemplos:
    python treinar.py                                  # treino padrão (yolo26n, 100 épocas)
    python treinar.py --modelo-base yolo26s.pt         # modelo maior, mais preciso
    python treinar.py --epocas 30                      # treino rápido para testar
    python treinar.py --dados meu_dataset/data.yaml    # seu próprio dataset (ex.: exportado do Roboflow)
    python treinar.py --retomar                        # continua um treino interrompido

No fim, o melhor modelo é copiado para  modelos/epi.pt  (usado pelo detectar_epi.py).
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


def escolher_dispositivo(pedido):
    if pedido:
        return pedido
    import torch
    if torch.cuda.is_available():  # também é True para AMD com PyTorch ROCm
        tipo = "AMD/ROCm" if getattr(torch.version, "hip", None) else "NVIDIA/CUDA"
        print(f"GPU encontrada ({tipo}): {torch.cuda.get_device_name(0)}")
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Usando GPU do Mac (MPS)")
        return "mps"
    print("\n[AVISO] Nenhuma GPU encontrada, treinando na CPU. Vai funcionar, mas é lento.\n"
          "        Para um teste rápido use --epocas 20. Para usar a GPU, instale o PyTorch\n"
          "        com CUDA (NVIDIA) ou ROCm (AMD) - veja o README.\n")
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="Treino do modelo de EPI com YOLO")
    ap.add_argument("--modelo-base", default="yolo26n.pt",
                    help="pesos iniciais: yolo26n.pt (rápido), yolo26s.pt, yolo26m.pt (mais preciso)")
    ap.add_argument("--dados", default="construction-ppe.yaml",
                    help="YAML do dataset (padrão: Construction-PPE, baixado automaticamente)")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="diminua (8, 4) se faltar memória")
    ap.add_argument("--paciencia", type=int, default=25, help="para antes se não melhorar por N épocas")
    ap.add_argument("--dispositivo", default=None, help="'cpu', '0', 'mps'. Padrão: automático")
    ap.add_argument("--workers", type=int, default=4, help="use 0 se der erro no Windows")
    ap.add_argument("--nome", default="epi", help="nome da pasta do treino em treinos/")
    ap.add_argument("--retomar", action="store_true", help="retoma o último treino com este --nome")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Instale as dependências primeiro:  pip install -r requirements.txt")

    dispositivo = escolher_dispositivo(args.dispositivo)
    pasta_treino = Path("treinos") / args.nome

    if args.retomar:
        ultimo = pasta_treino / "weights" / "last.pt"
        if not ultimo.exists():
            sys.exit(f"Não achei {ultimo} para retomar.")
        modelo = YOLO(str(ultimo))
        modelo.train(resume=True)
    else:
        modelo = YOLO(args.modelo_base)
        modelo.train(
            data=args.dados,
            epochs=args.epocas,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.paciencia,
            device=dispositivo,
            workers=args.workers,
            project=str(Path("treinos").resolve()),
            name=args.nome,
            exist_ok=True,
            plots=True,
        )

    melhor = Path(modelo.trainer.best) if getattr(modelo, "trainer", None) else pasta_treino / "weights" / "best.pt"
    if not melhor.exists():
        sys.exit("Treino terminou mas não encontrei o best.pt. Veja a pasta treinos/.")

    destino = Path("modelos") / "epi.pt"
    destino.parent.mkdir(exist_ok=True)
    if destino.exists():  # guarda o modelo anterior para poder voltar
        anterior = destino.with_name(f"epi_anterior_{time.strftime('%Y%m%d_%H%M%S')}.pt")
        shutil.copy(destino, anterior)
        print(f"Modelo anterior guardado em: {anterior}")
    shutil.copy(melhor, destino)
    print(f"\nModelo salvo em: {destino.resolve()}")

    # Validação final com métricas por classe
    final = YOLO(str(destino))
    m = final.val(data=args.dados, imgsz=args.imgsz, device=dispositivo, verbose=False)
    print("\n================ RESULTADO ================")
    print(f"mAP50     : {m.box.map50:.3f}")
    print(f"mAP50-95  : {m.box.map:.3f}")
    print("\nmAP50-95 por classe:")
    for i, nome in final.names.items():
        if i < len(m.box.maps):
            print(f"  {nome:<14} {m.box.maps[i]:.3f}")
    print("===========================================")
    print("\nAgora rode:  python detectar_epi.py")


if __name__ == "__main__":
    main()
