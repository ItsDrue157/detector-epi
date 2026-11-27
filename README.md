# Detector de EPI com YOLO (capacete e colete)

> **Guia rápido de uso:** veja o [COMO_USAR.md](COMO_USAR.md).

Detecta em tempo real, pela webcam (ou vídeo, foto, câmera IP), se cada pessoa está usando **capacete** e **colete**. Quem estiver sem EPI fica com a caixa vermelha. O programa também toca um bipe e salva uma foto mais uma linha em `ocorrencias/registro.csv`.

```
detector_epi/
├── detectar_epi.py        # detecção em tempo real (webcam/vídeo/foto/câmera IP)
├── treinar.py             # treina/re-treina o modelo (na GPU NVIDIA)
├── coletar_certo.py       # coleta fotos de EPI de verdade (rótulo de certo)
├── coletar_errado.py      # coleta fotos do que parece EPI mas não é (colete de futebol, boné)
├── coleta.py              # parte comum dos dois scripts de coleta
├── dados_treino.yaml      # dataset de treino: original + todas as coletas (gerado)
├── requirements.txt       # instalação só CPU
├── requirements-gpu.txt   # instalação com GPU NVIDIA (RTX série 40 ou 50)
├── modelos/
│   ├── epi.pt             # modelo de EPI atual
│   ├── epi_backup.pt      # modelo anterior (atualizado a cada treino)
│   ├── yolo26n.pt         # detector de pessoa leve (CPU)
│   └── yolo26s.pt         # detector de pessoa (GPU)
├── teste/                 # dataset original (Roboflow)
├── dados_bone/            # fotos de boné coletadas
├── dados_colete_certo/    # fotos de colete de EPI coletadas
├── dados_colete_errado/   # fotos de colete de futebol coletadas
└── tcc/                   # relatório e gráficos do re-treino
```

## Modelos

- `modelos/epi.pt` é um YOLO26s treinado no dataset brasileiro de EPI [anderson-vlsev/epi-d4clr](https://universe.roboflow.com/anderson-vlsev/epi-d4clr) (Roboflow, CC BY 4.0) e re-treinado com fotos de boné.
- `modelos/yolo26n.pt` e `modelos/yolo26s.pt` são os pesos COCO da Ultralytics. O modelo de EPI não tem classe de pessoa, então o detector usa um deles só para achar as pessoas: o `yolo26n` (mais leve) na CPU e o `yolo26s` na GPU.

| Classe do modelo | Uso no detector | Onde é procurada |
|---|---|---|
| `Capacete` | capacete | região da cabeça |
| `colete` | colete | região do tronco |
| `person` (de `yolo26n`/`yolo26s`) | pessoa | caixa de cada trabalhador |

O dataset também tem as classes `Balaclava`, `C_eletrica` e `Operacional`. Elas são filtradas já na inferência e não aparecem na tela.

## 1. Instalação

Use Python 3.10 a 3.12.

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt        # só CPU
pip install -r requirements-gpu.txt    # ou: GPU NVIDIA RTX série 40/50
```

O `requirements-gpu.txt` instala o PyTorch do índice CUDA 12.8, que é obrigatório para as RTX 50 e funciona nas RTX 40.

O `requirements.txt` instala o PyTorch **só de CPU**, de propósito. A versão com CUDA carrega bibliotecas que a CPU não usa: medido no modo CPU, o programa ocupa ~880 MB de RAM com o PyTorch CUDA e ~585 MB com o de CPU, na mesma velocidade. Por isso o PC da apresentação deve usar o `requirements.txt`, e não o de GPU.

## 2. Hardware

São dois modos, escolhidos pelo `--gpu`:

| | CPU (padrão) | GPU (`--gpu`) |
|---|---|---|
| Uso | apresentação, PC sem placa NVIDIA | PC com RTX 40/50, e treino |
| Imagem no modelo | 480 | 640 |
| Detector de pessoa | `yolo26n`, a cada 3 frames | `yolo26s`, todo frame |
| Tempo por frame | ~38 ms (~26 FPS, Intel 12ª geração) | ~25 ms (~40 FPS, RTX 5060) |
| mAP50 no teste | 0,833 | 0,834 |

Otimizações do modo CPU, todas medidas:

- **Imagem 480 em vez de 640:** 1,45× mais rápido no modelo de EPI, sem perda no mAP50. O mAP50-95 do colete cai um pouco (0,68 → 0,62).
- **Detector de pessoa nano a cada 3 frames:** 18 ms contra 48 ms do `yolo26s`, e achou 28 das 30 pessoas que o `yolo26s` achou. Pessoas mudam pouco entre frames, e o alerta já espera 8 frames seguidos.
- **Captura em thread:** webcam e câmera IP são lidas numa thread que guarda só o frame mais recente. Sem isso, com a CPU mais lenta que a câmera, a imagem vai ficando atrasada.
- Testado e descartado: ONNX Runtime na CPU ficou **mais lento** que o PyTorch (72 ms contra 48 ms em 640).

Se a placa não for RTX 40 ou 50, o `--gpu` avisa mas roda mesmo assim. Sem placa NVIDIA, ele cai para a CPU.

## 3. Rodar

```bash
python detectar_epi.py                               # webcam, CPU
python detectar_epi.py --gpu                         # webcam, GPU NVIDIA
python detectar_epi.py --fonte 1                     # outra webcam
python detectar_epi.py --fonte video.mp4 --salvar resultado.mp4
python detectar_epi.py --fonte foto.jpg              # gera foto_epi.jpg
python detectar_epi.py --fonte "rtsp://user:senha@192.168.0.10:554/stream"
python detectar_epi.py --exigir capacete             # cobra só capacete
python detectar_epi.py --sem-som
```

| Tecla | Ação |
|---|---|
| `q` / `ESC` | sair |
| `p` | pausar |
| `s` | salvar print em `ocorrencias/` |

Os ajustes finos ficam como constantes no topo do `detectar_epi.py` (seção **Configuração**): `CONF`, `CONF_CAPACETE`, `CONF_COLETE`, `IOU`, `LARGURA`/`ALTURA` da webcam, `INTERVALO_ALERTA` e `FRAMES_CONFIRMACAO`.

## Como a verificação funciona

1. O YOLO de EPI detecta capacetes e coletes, e o YOLO COCO detecta as pessoas.
2. Para cada pessoa, o programa procura um capacete na **região da cabeça** (topo da caixa) e um colete na **região do tronco**.
3. A pessoa fica **verde** se tiver tudo o que é exigido e **vermelha** se faltar algo.
4. O alerta (bipe, foto e CSV) só dispara quando a infração aparece em vários frames seguidos (`FRAMES_CONFIRMACAO`, padrão 8). Assim um frame ruim não gera alarme falso. Entre um alerta e outro há um intervalo de `INTERVALO_ALERTA` segundos (padrão 5).

As classes são reconhecidas **pelo nome** (helmet, hardhat, capacete, vest, colete, person, no_helmet, NO-Hardhat…), então outro modelo de EPI funciona trocando o `MODELO_EPI` no topo do arquivo.

## 4. Treinar

Treine no PC com a RTX. O `treinar.py` acha a GPU sozinho.

```bash
python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50
```

- O melhor modelo vai para `modelos/epi.pt` e o que estava lá vira `modelos/epi_backup.pt`. Fica sempre só um backup: o modelo anterior.
- No final aparecem as métricas por classe comparadas com o modelo base, e um relatório em `treinos/<nome>/relatorio.md`.
- Para voltar: `copy modelos\epi_backup.pt modelos\epi.pt`.

| Opção | Para quê |
|---|---|
| `--dados caminho/data.yaml` | dataset a usar (formato YOLO, ex.: exportado do Roboflow) |
| `--modelo-base modelos/epi.pt` | parte do modelo atual em vez de começar do zero |
| `--epocas 50` | número de épocas |
| `--batch 8` | se der erro de memória (CUDA out of memory) |
| `--workers 0` | se der erro de multiprocessing no Windows |
| `--retomar` | continua um treino que foi interrompido |

### Coletar fotos para o re-treino

```bash
python coletar_certo.py                    # colete de EPI de verdade  -> dados_colete_certo/
python coletar_errado.py                   # colete de futebol (falso) -> dados_colete_errado/
python coletar_errado.py --item capacete   # boné (falso capacete)     -> dados_bone/
python coletar_certo.py --fonte obra.mp4   # qualquer um aceita um vídeo gravado
```

Webcam: espaço salva, `a` salva sozinho a cada 1 s. Cada foto já sai rotulada com os EPIs que o modelo achou. A diferença é o item escolhido (`--item`, padrão colete):

- **`coletar_certo.py`**: o item é rotulado como EPI. Como ali só aparece EPI de verdade, detecções fracas (a partir de 0,15) também valem. Se o modelo não achar, `m` congela a imagem para desenhar as caixas com o mouse (`cv2.selectROIs`).
- **`coletar_errado.py`**: o item é descartado dos rótulos, e o modelo aprende que aquilo não é EPI. Não deixe aparecer o EPI de verdade.

A parte comum fica no `coleta.py`. A cada coleta, o `dados_treino.yaml` é refeito juntando o dataset original com **todas** as pastas de coleta, para um re-treino não fazer o modelo esquecer o que outra coleta ensinou. No fim do treino, o `treinar.py` mostra para cada coleta em quantas fotos o modelo ainda erra.

## Licença

O código e os modelos da Ultralytics usam a licença AGPL-3.0. O dataset de EPI é CC BY 4.0 (Roboflow, anderson-vlsev/epi-d4clr). Para uso comercial em software fechado, confira as condições de licença da Ultralytics.
