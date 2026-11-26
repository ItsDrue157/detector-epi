# Como usar o Detector de EPI

Guia prático, passo a passo. Para detalhes técnicos, veja o `README.md`.

---

## 1. Instalar (uma vez só)

Abra o terminal na pasta do projeto e crie o ambiente:

```bash
python -m venv venv
venv\Scripts\activate
```

Depois instale conforme o seu computador:

| Seu computador | Comando de instalação |
|---|---|
| **Placa NVIDIA** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` e depois `pip install -r requirements.txt` |
| **Placa AMD** (Windows 11 24H2 ou mais novo) | `pip install -r requirements-amd.txt` (num venv **novo**, sem o requirements.txt) |
| **Placa AMD** (Windows 10) | `pip install -r requirements-amd-win10.txt` (num venv **novo**, com Python 3.12) |
| **Sem placa de vídeo** (só CPU) | `pip install -r requirements.txt` |

> Toda vez que abrir um terminal novo, ative o ambiente com `venv\Scripts\activate` antes de rodar os comandos.

---

## 2. Rodar o detector

| Seu computador | Comando |
|---|---|
| Placa NVIDIA | `python detectar_epi.py` |
| Placa AMD | `python detectar_epi.py --dispositivo amd` |
| Só CPU | `python detectar_epi.py --dispositivo cpu` |

Na placa AMD, a primeira execução demora um pouco mais porque converte os modelos para ONNX. Depois fica rápido.

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

O modelo foi treinado também com as classes `Balaclava`, `C_eletrica` e `Operacional`, mas elas ficam desligadas e não aparecem na tela.

**Cores:** caixa **verde** = pessoa com todo o EPI. Caixa **vermelha** = falta capacete ou colete.

Quando alguém fica sem EPI por alguns segundos, o programa toca um bipe, salva uma foto e anota em `ocorrencias/registro.csv`.

---

## 3. Opções mais usadas

Os argumentos podem ir em **qualquer ordem**. Os que levam valor precisam do valor logo em seguida.

| Argumento | O que faz | Exemplo |
|---|---|---|
| `--dispositivo` | onde roda: `amd`, `cpu` ou `0` (NVIDIA) | `--dispositivo amd` |
| `--fonte` | câmera, vídeo, foto ou câmera IP | `--fonte 1`, `--fonte video.mp4` |
| `--sem-som` | desliga o bipe | `--sem-som` |
| `--exigir` | quais EPIs cobrar | `--exigir capacete` |
| `--conf` | confiança mínima geral (padrão 0.40) | `--conf 0.5` |
| `--conf-capacete` | confiança mínima só do capacete | `--conf-capacete 0.7` |
| `--conf-colete` | confiança mínima só do colete | `--conf-colete 0.6` |
| `--salvar` | grava o vídeo com as marcações | `--salvar resultado.mp4` |

Exemplo completo:

```bash
python detectar_epi.py --dispositivo amd --conf-capacete 0.7 --sem-som
```

> Use **ponto** nos números: `0.7` funciona, `0,7` dá erro.

---

## 4. O detector confunde boné com capacete

### Solução rápida: aumentar a confiança do capacete

1. Rode o detector e aponte para alguém **de boné**. Veja o número na caixa (ex.: `Capacete 0.52`).
2. Aponte para alguém **de capacete**. Veja o número (ex.: `Capacete 0.85`).
3. Use um valor no meio:

```bash
python detectar_epi.py --conf-capacete 0.7
```

Se o boné ainda passar, aumente (`0.75`, `0.8`). Se o capacete de verdade parar de ser reconhecido, diminua (`0.6`).

Se os dois números forem muito parecidos, essa solução não resolve. Faça o re-treino abaixo.

### Solução definitiva: re-treinar o modelo

Faça o treino num PC com **placa NVIDIA**. Na CPU ou na AMD com Windows ele leva horas.

**Passo 1: coletar fotos de boné**

```bash
python coletar_bones.py
```

- Coloque alguém **de boné** na frente da câmera, se mexendo: de frente, de lado, perto e longe.
- Aperte `a` para salvar uma foto por segundo automaticamente (ou `espaço` para salvar uma por vez). `q` sai.
- Na tela: em **verde** o que vai como rótulo (colete etc.). **Riscado em vermelho** o "capacete" que é boné, que será descartado.
- Junte **de 150 a 300 fotos**, com bonés de cores diferentes e pessoas diferentes, se der.
- Se preferir, grave um vídeo antes e use: `python coletar_bones.py --fonte video.mp4`

> ⚠️ **Não deixe aparecer capacete de verdade durante a coleta.** Ele seria ensinado como se não fosse capacete.

As fotos ficam na pasta `dados_bone/`. Dá para rodar o script várias vezes; ele vai somando as fotos.

**Passo 2: treinar**

```bash
python treinar.py --dados dados_bone/data.yaml --modelo-base modelos/epi_backup.pt --epocas 50
```

- Junta o dataset original (pasta `teste/`) com as suas fotos de boné.
- Leva uns 20 a 40 minutos numa placa NVIDIA.
- No final, o novo modelo vira o `modelos/epi.pt`. O anterior é guardado como `modelos/epi_anterior_<data>.pt`.

**Passo 3: testar**

```bash
python detectar_epi.py --sem-som
```

Se ainda confundir, colete mais fotos **justamente das situações em que errou** e treine de novo.

**Passo 4: levar para a máquina AMD (se for o caso)**

Copie o novo `modelos/epi.pt` para a máquina AMD e **apague os arquivos `modelos/*.onnx`** de lá. Na próxima execução ele converte o modelo novo sozinho.

### Voltar ao modelo antigo

Se o modelo novo ficou pior:

```bash
copy modelos\epi_backup.pt modelos\epi.pt
```

Na máquina AMD, apague também os `modelos/*.onnx`.

---

## 5. Problemas comuns

| Problema | Solução |
|---|---|
| `error: argument --conf-capacete: expected one argument` | Faltou o número: use `--conf-capacete 0.7` |
| Na AMD aparece `Using ONNX Runtime ... with CPUExecutionProvider` | Não achou a GPU. Atualize o driver AMD (Adrenalin) e o Windows. Rode `pip uninstall onnxruntime` (deixe só o `onnxruntime-windowsml`) |
| Webcam não abre | Tente `--fonte 1` ou `--fonte 2` |
| Muito lento | Use `--imgsz 480`, ou rode na GPU (`--dispositivo amd` ou NVIDIA) |
| Muitos alarmes falsos | Aumente `--conf` (ex.: `0.5`) ou `--frames-confirmacao` (ex.: `15`) |
| Erro de memória no treino (`CUDA out of memory`) | Adicione `--batch 8` ao comando do treino |
| Erro de multiprocessing no treino (Windows) | Adicione `--workers 0` ao comando do treino |
| Treino diz que não acha `train/images` | Confira se a pasta `teste/` tem `train/images` e `valid/images` |
