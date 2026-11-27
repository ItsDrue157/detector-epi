# Como usar o Detector de EPI

Guia prático, passo a passo. Para detalhes técnicos, veja o `README.md`.

---

## 1. Instalar (uma vez só)

Abra o terminal na pasta do projeto e crie o ambiente:

```bash
python -m venv venv
venv\Scripts\activate
```

Depois instale conforme o computador:

| Computador | Comando de instalação |
|---|---|
| **Só CPU** (PC da apresentação) | `pip install -r requirements.txt` |
| **Placa NVIDIA RTX série 40 ou 50** (4060, 5060...) | `pip install -r requirements-gpu.txt` |

> Toda vez que abrir um terminal novo, ative o ambiente com `venv\Scripts\activate` antes de rodar os comandos.

Para conferir se a placa NVIDIA foi reconhecida: `python -c "import torch; print(torch.cuda.is_available())"` tem que mostrar `True`.

---

## 2. Rodar o detector

| Computador | Comando |
|---|---|
| **Só CPU** (padrão) | `python detectar_epi.py` |
| **Placa NVIDIA** | `python detectar_epi.py --gpu` |

| | CPU | GPU |
|---|---|---|
| Tamanho da imagem no modelo | 480 | 640 |
| Detector de pessoa | `yolo26n.pt` (leve), a cada 3 frames | `yolo26s.pt`, em todo frame |
| FPS medido | ~26 (Intel 12ª geração) | ~40 (RTX 5060) |

No conjunto de teste, a precisão na CPU (mAP50 0,833) é praticamente a mesma da GPU (0,834).

**Teclas na janela:**

| Tecla | Ação |
|---|---|
| `q` ou `ESC` | sair |
| `p` | pausar / continuar |
| `s` | salvar um print em `ocorrencias/` |

**O que o detector verifica:**

| EPI | Onde procura |
|---|---|
| Capacete | na cabeça da pessoa |
| Colete | no tronco da pessoa |

**Cores:** caixa **verde** = pessoa com todo o EPI. Caixa **vermelha** = falta capacete ou colete.

Quando alguém fica sem EPI por alguns segundos, o programa toca um bipe, salva uma foto e anota em `ocorrencias/registro.csv`.

---

## 3. Opções

| Argumento | O que faz | Exemplo |
|---|---|---|
| `--fonte` | de onde vem a imagem: webcam, vídeo, foto ou câmera IP | `--fonte 1`, `--fonte video.mp4`, `--fonte foto.jpg`, `--fonte "rtsp://usuario:senha@192.168.0.10:554/stream"` |
| `--gpu` | roda na placa NVIDIA | `--gpu` |
| `--exigir` | quais EPIs cobrar | `--exigir capacete` |
| `--salvar` | grava o vídeo/foto com as marcações | `--salvar resultado.mp4` |
| `--sem-som` | desliga o bipe | `--sem-som` |

Exemplo:

```bash
python detectar_epi.py --fonte video.mp4 --salvar resultado.mp4 --sem-som
```

**Ajustes finos** (confiança mínima, confiança só do capacete ou do colete, tempo entre alertas, resolução da webcam) ficam no topo do `detectar_epi.py`, na seção **Configuração**. Edite o valor lá e rode de novo.

---

## 4. Re-treinar o modelo (ex.: boné, colete)

Faça o treino **no PC com a RTX**. Na CPU ele leva horas.

**Modelos na pasta `modelos/`:**

| Arquivo | O que é |
|---|---|
| `epi.pt` | modelo atual (usado pelo detector) |
| `epi_backup.pt` | o modelo anterior. A cada treino, o `epi.pt` antigo vira o novo backup |
| `yolo26n.pt`, `yolo26s.pt` | detectores de pessoa (não apagar) |

**Passo 1: coletar fotos de colete de EPI**

```bash
python coletar_colete.py                    # webcam
python coletar_colete.py --fonte video.mp4  # ou um vídeo gravado
```

As fotos vão para `dados_colete/`.

- Coloque alguém **de colete de EPI** na frente da câmera, se mexendo: de frente, de lado, de costas, perto e longe.
- Aperte `a` para salvar uma foto por segundo automaticamente (ou `espaço` para salvar uma por vez). `q` sai.
- Em **verde grosso** aparece o colete que vai ser rotulado. Como ali só tem colete de verdade, ele aceita até detecções fracas.
- Se aparecer **"colete não encontrado: aperte M"**, aperte `m`: a imagem congela. Arraste uma caixa sobre cada colete, aperte `ENTER` a cada caixa e `ESC` para salvar. Essas fotos são as que mais ensinam o modelo.
- No automático, ele só salva as fotos em que achou o colete. As outras, marque com `m`.
- Junte **de 150 a 300 fotos**, com coletes e pessoas diferentes, se der.

> ⚠️ **Só colete de EPI de verdade.** Nada de colete de futebol nessa coleta.

**Passo 2: treinar, partindo do modelo atual**

```bash
python treinar.py --dados dados_treino.yaml --modelo-base modelos/epi.pt --epocas 50
```

- O `dados_treino.yaml` é atualizado sozinho pelo `coletar_colete.py` e junta o dataset original, as fotos de boné e as de colete. Assim o modelo aprende o colete sem esquecer o boné.
- Leva uns 20 a 40 minutos na RTX.
- No final aparece, para cada coleta, em quantas fotos o modelo ainda erra, antes e depois do treino: colete de futebol marcado como colete, e colete de EPI não encontrado.
- O novo modelo vira o `modelos/epi.pt` e o anterior vira o `modelos/epi_backup.pt`.

**Passo 3: testar e levar para o PC da apresentação**

```bash
python detectar_epi.py --gpu --sem-som
```

Se ficou bom, copie o `modelos/epi.pt` para o PC da apresentação (ou faça `git pull` lá).

### Voltar ao modelo anterior

```bash
copy modelos\epi_backup.pt modelos\epi.pt
```

---

## 5. Problemas comuns

| Problema | Solução |
|---|---|
| Webcam não abre | Tente `--fonte 1` ou `--fonte 2`. Feche outros programas que usam a câmera |
| Câmera IP não abre | Teste a URL no VLC primeiro (Mídia → Abrir fluxo de rede). Coloque a URL entre aspas |
| `--gpu` diz que não achou placa NVIDIA | Instale com `pip install -r requirements-gpu.txt` num venv novo e atualize o driver da NVIDIA |
| Muito lento na CPU | Feche navegador e outros programas pesados. Use o notebook na tomada (na bateria a CPU fica mais lenta) |
| Aparece `'modelos/yolo26n.pt' não encontrado` | Atualize o projeto (`git pull`). Sem ele o programa usa o `yolo26s.pt`, que é mais lento |
| Muitos alarmes falsos | Aumente `CONF` (ex.: `0.5`) ou `FRAMES_CONFIRMACAO` (ex.: `15`) no topo do `detectar_epi.py` |
| Erro de memória no treino (`CUDA out of memory`) | Adicione `--batch 8` ao comando do treino |
| Erro de multiprocessing no treino (Windows) | Adicione `--workers 0` ao comando do treino |
| Treino diz que não acha `train/images` | Confira se a pasta `teste/` tem `train/images` e `valid/images` |
