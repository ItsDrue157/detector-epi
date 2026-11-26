"""
Treina o modelo YOLO de detecção de EPI, com painel de progresso e métricas.

Por padrão usa o dataset aberto "Construction-PPE" da Ultralytics, que é
baixado automaticamente na primeira execução (capacete, colete, luvas,
botas, óculos, pessoa e as classes de "sem EPI").

Exemplos:
    python treinar.py                                  # treino padrão (yolo26n, 100 épocas)
    python treinar.py --modelo-base yolo26s.pt         # modelo maior, mais preciso
    python treinar.py --epocas 30                      # treino rápido para testar
    python treinar.py --dados meu_dataset/data.yaml    # seu próprio dataset (ex.: exportado do Roboflow)
    python treinar.py --retomar                        # continua um treino interrompido
    python treinar.py --log-completo                   # mostra também o log original do YOLO

    # re-treino com as fotos de boné (coletar_bones.py):
    python treinar.py --dados dados_bone/data.yaml --modelo-base modelos/epi_backup.pt --epocas 50

No fim:
  - o melhor modelo é copiado para  modelos/epi.pt  (usado pelo detectar_epi.py);
  - aparecem as métricas por classe, comparadas com o modelo base;
  - se houver fotos de boné, mostra quantas ainda são confundidas com capacete;
  - um relatório é salvo em  treinos/<nome>/relatorio.md
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Silencia o log padrão do YOLO (precisa vir antes de importar o ultralytics).
if "--log-completo" not in sys.argv:
    os.environ.setdefault("YOLO_VERBOSE", "False")

from detectar_epi import EXT_IMAGEM, categoria_da_classe  # noqa: E402

# --------------------------------------------------------------------------- #
# Visual do terminal
# --------------------------------------------------------------------------- #
if os.name == "nt":
    os.system("")  # liga as cores ANSI no console do Windows
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

_COR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(codigo):
    return (lambda t: f"\033[{codigo}m{t}\033[0m") if _COR else (lambda t: str(t))


negrito, fraco = _c("1"), _c("2")
verde, amarelo, vermelho, ciano, magenta = _c("32"), _c("33"), _c("31"), _c("36"), _c("35")
LARG = 76


def titulo(texto):
    print()
    print(ciano("╭" + "─" * (LARG - 2) + "╮"))
    print(ciano("│") + negrito(f" {texto}".ljust(LARG - 2)) + ciano("│"))
    print(ciano("╰" + "─" * (LARG - 2) + "╯"))


def linha_info(rotulo, valor):
    print(f"  {fraco(rotulo.ljust(16))} {valor}")


def barra(valor, largura=20):
    valor = max(0.0, min(1.0, float(valor)))
    cheio = int(round(valor * largura))
    cor = verde if valor >= 0.85 else amarelo if valor >= 0.6 else vermelho
    return cor("█" * cheio) + fraco("░" * (largura - cheio))


def cor_valor(v):
    t = f"{v:.3f}"
    return verde(t) if v >= 0.85 else amarelo(t) if v >= 0.6 else vermelho(t)


def delta(novo, antigo):
    if antigo is None:
        return fraco("   -  ")
    d = novo - antigo
    t = f"{d:+.3f}"
    if abs(d) < 0.005:
        return fraco(t)
    return verde(t) if d > 0 else vermelho(t)


def duracao(seg):
    seg = int(seg)
    h, r = divmod(seg, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# Dispositivo
# --------------------------------------------------------------------------- #
def escolher_dispositivo(pedido):
    import torch
    if pedido:
        return pedido, pedido
    if torch.cuda.is_available():  # também é True para AMD com PyTorch ROCm
        tipo = "AMD/ROCm" if getattr(torch.version, "hip", None) else "NVIDIA/CUDA"
        return "0", f"{torch.cuda.get_device_name(0)} ({tipo})"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "GPU do Mac (MPS)"
    print(amarelo("\n[AVISO] Nenhuma GPU encontrada, treinando na CPU. Vai funcionar, mas é lento.\n"
                  "        Para um teste rápido use --epocas 20. Para usar a GPU, instale o PyTorch\n"
                  "        com CUDA (NVIDIA) ou ROCm (AMD) - veja o README."))
    return "cpu", "CPU"


# --------------------------------------------------------------------------- #
# Progresso do treino (callbacks do YOLO)
# --------------------------------------------------------------------------- #
class Painel:
    """Barra de progresso por época e uma linha de resumo ao fim de cada uma."""

    def __init__(self, silencioso):
        self.silencioso = silencioso  # se False, o log original do YOLO também aparece
        self.t0 = self.t_epoca = time.time()
        self.lote = 0
        self.melhor = None
        self.sem_melhorar = 0
        self.historico = []

    def registrar(self, modelo):
        for evento, fn in (("on_train_start", self.inicio), ("on_train_epoch_start", self.epoca_inicio),
                           ("on_train_batch_end", self.lote_fim), ("on_train_epoch_end", self.validando),
                           ("on_fit_epoch_end", self.epoca_fim)):
            modelo.add_callback(evento, self._seguro(fn))

    @staticmethod
    def _seguro(fn):
        def f(trainer):
            try:
                fn(trainer)
            except Exception as e:  # o painel nunca pode derrubar o treino
                print(fraco(f"\n  (painel: {e})"))
        return f

    def inicio(self, tr):
        self.t0 = time.time()
        n_treino = len(tr.train_loader.dataset)
        n_val = len(tr.test_loader.dataset) if getattr(tr, "test_loader", None) else 0
        titulo("Treinando")
        linha_info("Imagens", f"{negrito(n_treino)} treino · {negrito(n_val)} validação")
        linha_info("Lotes", f"{len(tr.train_loader)} por época · lote de {tr.batch_size}")
        linha_info("Épocas", f"até {tr.epochs} (para antes se passar {tr.args.patience} sem melhorar)")
        print()
        print(fraco("  época      loss   precisão  recall   mAP50  mAP50-95   tempo   faltam"))
        print(fraco("  " + "─" * (LARG - 4)))

    def epoca_inicio(self, tr):
        self.t_epoca = time.time()
        self.lote = 0

    def _loss(self, tr):
        tl = getattr(tr, "tloss", None)
        if tl is None:
            return float("nan")
        valores = tl.values() if isinstance(tl, dict) else tl  # dict nas versões novas do YOLO
        return float(sum(float(v.sum()) if hasattr(v, "sum") else float(v) for v in valores))

    def lote_fim(self, tr):
        self.lote += 1
        if not self.silencioso:
            return
        total = len(tr.train_loader)
        frac = self.lote / max(total, 1)
        n = int(frac * 24)
        print(f"\r  {tr.epoch + 1:>3}/{tr.epochs:<4} {ciano('━' * n)}{fraco('─' * (24 - n))} "
              f"{frac:4.0%}  loss {self._loss(tr):.3f}  {duracao(time.time() - self.t_epoca)}   ",
              end="", flush=True)

    def validando(self, tr):
        if self.silencioso:
            print(f"\r  {tr.epoch + 1:>3}/{tr.epochs:<4} {fraco('validando...'):<60}", end="", flush=True)

    def epoca_fim(self, tr):
        if self.historico and self.lote == 0:
            return  # validação final do best.pt dispara o evento de novo, sem treinar: ignora
        feitas = tr.epoch + 1
        m = tr.metrics or {}
        p = m.get("metrics/precision(B)", 0.0)
        r = m.get("metrics/recall(B)", 0.0)
        m50 = m.get("metrics/mAP50(B)", 0.0)
        m5095 = m.get("metrics/mAP50-95(B)", 0.0)
        fit = tr.fitness if tr.fitness is not None else m5095

        novo_melhor = self.melhor is None or fit > self.melhor
        if novo_melhor:
            self.melhor, self.sem_melhorar = fit, 0
        else:
            self.sem_melhorar += 1

        agora = time.time()
        t_ep = agora - self.t_epoca
        media = (agora - self.t0) / feitas
        faltam = media * (tr.epochs - feitas)
        self.historico.append((feitas, self._loss(tr), p, r, m50, m5095))

        self.lote = 0
        marca = verde(" ★ melhor") if novo_melhor else fraco(f"  ({self.sem_melhorar} sem melhorar)")
        print(f"\r  {feitas:>3}/{tr.epochs:<4} {self._loss(tr):7.3f}   {cor_valor(p)}   {cor_valor(r)}   "
              f"{cor_valor(m50)}   {cor_valor(m5095)}   {duracao(t_ep):>6}  {duracao(faltam):>6}{marca}")


# --------------------------------------------------------------------------- #
# Métricas finais
# --------------------------------------------------------------------------- #
def validar(caminho, dados, imgsz, dispositivo, pasta):
    from ultralytics import YOLO
    modelo = YOLO(str(caminho))
    m = modelo.val(data=dados, imgsz=imgsz, device=dispositivo, verbose=False, plots=False,
                   project=str(pasta.resolve()), name="val", exist_ok=True)
    por_classe = {r["Class"]: r for r in m.summary()}
    geral = {
        "P": m.box.mp, "R": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map,
    }
    return modelo.names, geral, por_classe


def contar_bones(caminho, pasta_imgs, dispositivo, conf=0.40):
    """Quantas fotos de boné (que não têm capacete) o modelo ainda marca como capacete."""
    from ultralytics import YOLO
    imgs = sorted(p for p in Path(pasta_imgs).iterdir() if p.suffix.lower() in EXT_IMAGEM)
    if not imgs:
        return None
    modelo = YOLO(str(caminho))
    cls_capacete = [i for i, n in modelo.names.items() if categoria_da_classe(n) == "capacete"]
    if not cls_capacete:
        return None
    erradas = 0
    for r in modelo.predict([str(p) for p in imgs], conf=conf, classes=cls_capacete, device=dispositivo,
                            verbose=False, stream=True):
        erradas += int(r.boxes is not None and len(r.boxes) > 0)
    return erradas, len(imgs)


def tabela_classes(nomes, por_classe, antes):
    print(fraco("  classe          objetos  precisão  recall   mAP50   mAP50-95   vs. antes"))
    print(fraco("  " + "─" * (LARG - 4)))
    linhas_md = []
    for _, nome in sorted(nomes.items()):
        r = por_classe.get(nome)
        if r is None:
            continue
        usada = categoria_da_classe(nome) is not None
        ant = antes.get(nome, {}).get("mAP50-95") if antes else None
        rotulo = negrito(f"{nome:<14}") if usada else fraco(f"{nome:<14}")
        valores = (f"{r['Instances']:>7}   {cor_valor(r['Box-P'])}   {cor_valor(r['Box-R'])}   "
                   f"{cor_valor(r['mAP50'])}   {cor_valor(r['mAP50-95'])}    {delta(r['mAP50-95'], ant)}")
        print(f"  {rotulo}  {valores if usada else fraco(_sem_cor(valores))}")
        d = f"{r['mAP50-95'] - ant:+.3f}" if ant is not None else "-"
        linhas_md.append(f"| {nome}{' (usada)' if usada else ''} | {r['Instances']} | {r['Box-P']:.3f} | "
                         f"{r['Box-R']:.3f} | {r['mAP50']:.3f} | {r['mAP50-95']:.3f} | {d} |")
    return linhas_md


def _sem_cor(texto):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", texto)


def relatorio_md(caminho, args, disp_nome, duracao_total, geral, geral_antes, linhas_md, bones,
                 pasta_treino):
    def lin(chave, nome):
        a = f"{geral_antes[chave]:.3f}" if geral_antes else "-"
        d = f"{geral[chave] - geral_antes[chave]:+.3f}" if geral_antes else "-"
        return f"| {nome} | {a} | {geral[chave]:.3f} | {d} |"

    texto = [
        "# Relatório de treino",
        "",
        f"- Data: {time.strftime('%d/%m/%Y %H:%M')}",
        f"- Dataset: `{args.dados}`",
        f"- Modelo base: `{args.modelo_base}`",
        f"- Épocas pedidas: {args.epocas} · imgsz {args.imgsz} · lote {args.batch}",
        f"- Dispositivo: {disp_nome}",
        f"- Duração: {duracao_total}",
        "",
        "## Métricas gerais (validação)",
        "",
        "| Métrica | Antes | Depois | Diferença |",
        "|---|---|---|---|",
        lin("P", "Precisão"),
        lin("R", "Recall"),
        lin("mAP50", "mAP50"),
        lin("mAP50-95", "mAP50-95"),
        "",
        "## Por classe",
        "",
        "| Classe | Objetos | Precisão | Recall | mAP50 | mAP50-95 | Δ mAP50-95 |",
        "|---|---|---|---|---|---|---|",
        *linhas_md,
    ]
    if bones:
        texto += ["", "## Boné confundido com capacete (fotos de validação)", ""]
        if bones.get("antes"):
            e, n = bones["antes"]
            texto.append(f"- Antes: {e} de {n} fotos ({e / n:.0%})")
        e, n = bones["depois"]
        texto.append(f"- Depois: {e} de {n} fotos ({e / n:.0%})")
    texto += [
        "",
        "## Gráficos",
        "",
        "Na pasta do treino:",
        "",
        "- `results.png`: curvas de loss e métricas por época",
        "- `confusion_matrix.png`: o que o modelo confunde com o quê",
        "- `BoxPR_curve.png`: precisão x recall por classe",
        "- `val_batch0_pred.jpg`: exemplos de detecção na validação",
        "",
        "## O que cada métrica significa",
        "",
        "- **Precisão**: das caixas que o modelo desenhou, quantas estavam certas (poucos alarmes falsos).",
        "- **Recall**: dos EPIs que existiam nas fotos, quantos o modelo achou (poucos EPIs perdidos).",
        "- **mAP50**: qualidade geral da detecção, aceitando caixas com 50% de sobreposição.",
        "- **mAP50-95**: igual, mas exige caixas mais justas. É a métrica mais rígida.",
        "",
    ]
    caminho.write_text("\n".join(texto), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
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
    ap.add_argument("--log-completo", action="store_true", help="mostra também o log original do YOLO")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Instale as dependências primeiro:  pip install -r requirements.txt")

    dispositivo, disp_nome = escolher_dispositivo(args.dispositivo)
    pasta_treino = Path("treinos") / args.nome
    base = Path(args.modelo_base)

    titulo("Detector de EPI · Treino do modelo")
    linha_info("Dataset", args.dados)
    linha_info("Modelo base", args.modelo_base if not args.retomar else f"retomando {pasta_treino}")
    linha_info("Dispositivo", disp_nome)
    linha_info("Imagem", f"{args.imgsz}px · lote {args.batch}")
    linha_info("Saída", f"{pasta_treino}/  →  modelos/epi.pt")

    painel = Painel(silencioso=not args.log_completo)
    t_inicio = time.time()
    try:
        if args.retomar:
            ultimo = pasta_treino / "weights" / "last.pt"
            if not ultimo.exists():
                sys.exit(f"Não achei {ultimo} para retomar.")
            modelo = YOLO(str(ultimo))
            painel.registrar(modelo)
            modelo.train(resume=True)
        else:
            modelo = YOLO(args.modelo_base)
            painel.registrar(modelo)
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
    except KeyboardInterrupt:
        print(amarelo("\n\nTreino interrompido. Para continuar de onde parou:"))
        print(f"  python treinar.py --nome {args.nome} --retomar")
        sys.exit(1)
    duracao_total = duracao(time.time() - t_inicio)

    melhor = Path(modelo.trainer.best) if getattr(modelo, "trainer", None) else pasta_treino / "weights" / "best.pt"
    if not melhor.exists():
        sys.exit("Treino terminou mas não encontrei o best.pt. Veja a pasta treinos/.")

    # ----- avaliação: modelo base (antes) x modelo novo (depois) ------------ #
    titulo("Avaliando")
    print(fraco("  validando o modelo novo..."))
    nomes, geral, por_classe = validar(melhor, args.dados, args.imgsz, dispositivo, pasta_treino / "_avaliacao")

    geral_antes, antes = None, None
    if base.exists() and not args.retomar:
        try:
            nomes_base = YOLO(str(base)).names
            if dict(nomes_base) == dict(nomes):
                print(fraco(f"  validando o modelo base ({base.name}) para comparar..."))
                _, geral_antes, antes = validar(base, args.dados, args.imgsz, dispositivo,
                                                pasta_treino / "_avaliacao_base")
        except Exception as e:
            print(fraco(f"  (não deu para validar o modelo base: {e})"))

    bones = None
    pasta_bones = Path(args.dados).parent / "images" / "val"
    if pasta_bones.is_dir():
        print(fraco("  testando fotos de boné..."))
        bones = {"depois": contar_bones(melhor, pasta_bones, dispositivo)}
        if geral_antes is not None:
            bones["antes"] = contar_bones(base, pasta_bones, dispositivo)
        if not bones["depois"]:
            bones = None

    # ----- copia o modelo para modelos/epi.pt ------------------------------- #
    destino = Path("modelos") / "epi.pt"
    destino.parent.mkdir(exist_ok=True)
    anterior = None
    if destino.exists():  # guarda o modelo anterior para poder voltar
        anterior = destino.with_name(f"epi_anterior_{time.strftime('%Y%m%d_%H%M%S')}.pt")
        shutil.copy(destino, anterior)
    shutil.copy(melhor, destino)

    # ----- resultado -------------------------------------------------------- #
    titulo(f"Resultado · treino concluído em {duracao_total}")

    print(negrito("  Resumo geral") + fraco("  (validação)"))
    print()
    for chave, rotulo in (("P", "Precisão"), ("R", "Recall"), ("mAP50", "mAP50"), ("mAP50-95", "mAP50-95")):
        ant = geral_antes[chave] if geral_antes else None
        extra = f"   antes {ant:.3f}  {delta(geral[chave], ant)}" if ant is not None else ""
        print(f"  {rotulo:<10} {barra(geral[chave])}  {cor_valor(geral[chave])}{extra}")

    print()
    print(negrito("  Por classe") + fraco("  (em negrito: classes usadas pelo detector)"))
    print()
    linhas_md = tabela_classes(nomes, por_classe, antes)

    if bones:
        print()
        print(negrito("  Boné confundido com capacete") + fraco("  (fotos de boné da validação)"))
        print()
        e, n = bones["depois"]
        if bones.get("antes"):
            ea, na = bones["antes"]
            print(f"  antes   {barra(1 - ea / na)}  {ea} de {na} fotos marcadas como capacete")
        print(f"  depois  {barra(1 - e / n)}  {negrito(e)} de {n} fotos marcadas como capacete")

    rel = pasta_treino / "relatorio.md"
    relatorio_md(rel, args, disp_nome, duracao_total, geral, geral_antes, linhas_md, bones, pasta_treino)

    print()
    print(fraco("  " + "─" * (LARG - 4)))
    linha_info("Modelo novo", verde(str(destino.resolve())))
    if anterior:
        linha_info("Modelo antigo", str(anterior))
    linha_info("Relatório", str(rel.resolve()))
    linha_info("Gráficos", str(pasta_treino.resolve()))
    print()
    print(fraco("  Precisão = poucos alarmes falsos · Recall = poucos EPIs perdidos · "
                "mAP = qualidade geral"))
    print()
    print(f"  Agora rode:  {negrito('python detectar_epi.py')}")
    print()


if __name__ == "__main__":
    main()
