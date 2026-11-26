# Detector de EPI com YOLO (capacete e colete)

Detecta em tempo real, pela webcam (ou vídeo, imagem, câmera IP), se cada pessoa está usando **capacete** e **colete**. Quem estiver sem EPI fica com a caixa vermelha. O programa também toca um bipe e salva uma foto mais uma linha em `ocorrencias/registro.csv`.

```
detector_epi/
├── detectar_epi.py    # detecção em tempo real (webcam/vídeo/imagem)
├── baixar_modelo.py   # baixa um modelo de EPI JÁ TREINADO (mais fácil)
├── treinar.py         # OU treina o seu próprio modelo
├── requirements.txt
└── modelos/epi.pt     # criado pelo baixar_modelo.py ou pelo treinar.py
```

## Jeito mais rápido (modelo já incluído)

O repositório já vem com o modelo pronto:

- `modelos/epi.pt` é um YOLO26s treinado no dataset brasileiro de EPI [anderson-vlsev/epi-d4clr](https://universe.roboflow.com/anderson-vlsev/epi-d4clr) (Roboflow, CC BY 4.0). Classes: `Balaclava`, `C_eletrica`, `Capacete`, `Operacional` e `colete`.
- `modelos/yolo26s.pt` são os pesos COCO da Ultralytics. O modelo de EPI não tem classe de pessoa, então o script usa este modelo só para achar as pessoas (`--modelo-pessoa`).

```bash
pip install -r requirements.txt
python detectar_epi.py       # abre a webcam
```

> **Atenção:** o `baixar_modelo.py` e o `treinar.py` **substituem** o `modelos/epi.pt`.

## 1. Instalação

Use Python 3.9 a 3.12.

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

**Tem placa NVIDIA?** Instale o PyTorch com CUDA *antes* do requirements. O treino fica umas 20× mais rápido.
Pegue o comando em https://pytorch.org/get-started/locally/. Exemplo:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Sem placa NVIDIA: GPU AMD ou só CPU

**Só CPU (qualquer PC):** o `pip install -r requirements.txt` já instala o PyTorch para CPU. Rode com:

```bash
python detectar_epi.py --dispositivo cpu
```

Funciona igual, só que com menos FPS. Para ganhar velocidade na CPU, use `--imgsz 480` ou veja a dica de OpenVINO no final.

**GPU AMD no Windows 11 (24H2 ou mais novo):** use o Windows ML, que roda o modelo em ONNX na placa AMD (também serve para Intel Arc). Num venv **limpo**:

```bash
pip install -r requirements-amd.txt
python detectar_epi.py --dispositivo amd
```

- Na primeira execução os modelos `.pt` são convertidos para `modelos/epi_640.onnx` e `modelos/yolo26s_640.onnx`. Depois disso a conversão não se repete. Se mudar o `--imgsz`, ele gera outro arquivo.
- No início aparece `Using ONNX Runtime ... with ...`, que mostra onde o modelo está rodando. Se aparecer `CPUExecutionProvider`, ele não achou a GPU: atualize o driver da AMD (Adrenalin) e o Windows, e confira se o pacote `onnxruntime` comum **não** está instalado (`pip uninstall onnxruntime`).
- Em Windows mais antigo que o 24H2, o Windows ML não funciona e o programa roda na CPU.

**GPU AMD no Linux (ROCm):** instale o PyTorch com ROCm antes do requirements (comando em https://pytorch.org/get-started/locally/, opção ROCm) e rode normalmente com `--dispositivo 0`. Nesse caso o treino (`treinar.py`) também usa a GPU AMD.

> O treino (`treinar.py`) precisa de PyTorch com CUDA ou ROCm para usar a GPU. Numa AMD no Windows ele roda na CPU, e é bem lento. Uma alternativa é treinar no Google Colab e copiar o `epi.pt` para a pasta `modelos/`.

## 2. Conseguir o modelo

### Opção A: baixar um modelo pronto (recomendado para começar)

```bash
python baixar_modelo.py
```

Ele baixa o **SafetyVision YOLOv8s**, treinado por terceiros em cerca de 80 mil imagens de obras e indústrias (https://huggingface.co/ayushgupta7777/safetyvision-yolov8). Detecta Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest e Person, entre outras classes. Segundo o autor, capacete é a classe em que ele vai melhor. A detecção direta de "sem colete" é a mais fraca, mas o `detectar_epi.py` não depende dela: ele marca "SEM COLETE" quando não encontra um colete no tronco da pessoa.

Se o download pelo script falhar, baixe pelo navegador o arquivo
https://huggingface.co/ayushgupta7777/safetyvision-yolov8/resolve/main/v2/best.pt
e salve como `modelos/epi.pt`.

### Opção B: treinar o seu (uma vez só)

```bash
python treinar.py
```

- Na primeira execução, o dataset aberto **Construction-PPE** da Ultralytics é baixado sozinho (cerca de 180 MB), junto com os pesos iniciais `yolo26n.pt`.
- O melhor modelo vai para `modelos/epi.pt`. No final aparecem as métricas (mAP) de cada classe.
- Tempo aproximado: de 15 a 40 min numa GPU comum. Na CPU são várias horas. Para testar rápido, use `python treinar.py --epocas 20`.

Opções úteis:

| Comando | Para quê |
|---|---|
| `--modelo-base yolo26s.pt` | modelo maior, mais preciso e um pouco mais lento |
| `--batch 8` | se der erro de memória (CUDA out of memory) |
| `--workers 0` | se der erro de multiprocessing no Windows |
| `--retomar` | continua um treino que foi interrompido |
| `--dados caminho/data.yaml` | treina com o **seu** dataset (ex.: exportado do Roboflow em formato YOLO) |

## 3. Rodar na webcam

```bash
python detectar_epi.py
```

| Tecla | Ação |
|---|---|
| `q` / `ESC` | sair |
| `p` | pausar |
| `s` | salvar print em `ocorrencias/` |

Outros exemplos:

```bash
python detectar_epi.py --fonte 1                     # outra webcam
python detectar_epi.py --fonte video.mp4 --salvar resultado.mp4
python detectar_epi.py --fonte foto.jpg              # gera foto_epi.jpg
python detectar_epi.py --fonte "rtsp://user:senha@192.168.0.10:554/stream"
python detectar_epi.py --exigir capacete             # cobra só capacete
python detectar_epi.py --conf 0.5                    # menos falsos positivos
python detectar_epi.py --sem-som
python detectar_epi.py --dispositivo amd             # GPU AMD (Windows ML)
python detectar_epi.py --dispositivo cpu             # só processador
```

## Como a verificação funciona

1. O YOLO detecta pessoas, capacetes, coletes e também as classes de "sem capacete" e "sem colete" que o dataset tiver.
2. Para cada pessoa, o programa procura um capacete na **região da cabeça** (topo da caixa) e um colete na **região do tronco**.
3. A pessoa fica **verde** se tiver tudo o que é exigido e **vermelha** se faltar algo, ou se o modelo detectar diretamente "sem capacete"/"sem colete" nela.
4. O alerta (bipe, foto e CSV) só dispara quando a infração aparece em vários frames seguidos (`--frames-confirmacao`, padrão 8). Assim um frame ruim não gera alarme falso. Entre um alerta e outro há um intervalo de `--intervalo-alerta` segundos.

As classes são reconhecidas **pelo nome** (helmet, hardhat, hard_hat, vest, jacket, person, worker, no_helmet, NO-Hardhat, NO-Safety Vest, helmet_off…). Por isso você pode usar outro modelo de EPI pronto com `--modelo outro_modelo.pt`, sem mexer no código. Ao abrir, o programa mostra como mapeou cada classe.

## Dicas para melhorar a precisão

- **Boné ou chapéu sendo aceito como capacete:** veja o número que aparece na caixa do boné e na de um capacete de verdade, e use um `--conf-capacete` entre os dois (ex.: `--conf-capacete 0.7`). O mesmo vale para `--conf-colete` com camisas coloridas. Se os números forem parecidos, só re-treinando com fotos de boné resolve (veja abaixo).
- Teste com as câmeras e a iluminação reais do seu local. Se errar muito, grave alguns minutos, rotule cerca de 200 a 500 imagens (Roboflow ou CVAT, formato YOLO) e treine com `--dados`.
- Pessoas muito pequenas na imagem: use `--imgsz 960` no treino e na detecção, ou um modelo maior (`yolo26s.pt`/`yolo26m.pt`).
- Para rodar mais rápido na CPU, exporte para ONNX/OpenVINO:
  `yolo export model=modelos/epi.pt format=openvino` e depois `--modelo modelos/epi_openvino_model`.

## Licença do dataset

O código e os modelos da Ultralytics usam a licença AGPL-3.0, e o dataset Construction-PPE tem licença própria (veja em docs.ultralytics.com/datasets/detect/construction-ppe). Para uso comercial em software fechado, confira as condições de licença da Ultralytics.
