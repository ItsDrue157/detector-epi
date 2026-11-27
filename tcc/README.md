# Material para o TCC

Resultados e evidências dos experimentos do Detector de EPI, prontos para usar no texto, nas figuras e na apresentação.

## `retreino_bone/`: re-treino para não confundir boné com capacete

Treino feito em 26/11/2026, numa NVIDIA RTX 5060. Levou 17 minutos e parou na época 39 de 50 por *early stopping* (o melhor resultado foi o da época 14).

| Arquivo / pasta | O que é | Uso sugerido no TCC |
|---|---|---|
| `index.html` | Relatório visual completo (abra no navegador) | Apresentação, visão geral |
| `relatorio.md` | Relatório gerado pelo `treinar.py`, com as tabelas de métricas | Copiar tabelas para o texto |
| `comparacoes/` | Mesmas imagens com o modelo antigo (ANTES) e o novo (DEPOIS), lado a lado | Figuras de resultado qualitativo |
| `graficos/results.png` | Loss e métricas por época | Figura da curva de treino |
| `graficos/confusion_matrix*.png` | Matriz de confusão (absoluta e normalizada) | Análise de erros por classe |
| `graficos/BoxPR_curve.png` | Curva precisão x recall por classe | Discussão de desempenho |
| `graficos/BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` | F1, precisão e recall por limiar de confiança | Justificar o limiar de confiança |
| `graficos/labels.jpg` | Distribuição das anotações no dataset | Descrição do dataset |
| `graficos/train_batch*.jpg` | Exemplos de lotes de treino com *data augmentation* | Metodologia |
| `graficos/val_batch*_labels.jpg` / `_pred.jpg` | Validação: anotação real x predição do modelo | Resultado qualitativo |
| `dados/results.csv` | Métricas de cada época (números brutos) | Refazer gráficos, tabelas |
| `dados/args.yaml` | Todos os hiperparâmetros usados no treino | Seção de metodologia (reprodutibilidade) |
| `img/` | Versões menores dos gráficos, usadas pelo `index.html` | - |

### Principais resultados

| Métrica | Antes | Depois |
|---|---|---|
| Fotos de boné marcadas como capacete (47 de validação) | 14 (30%) | **0 (0%)** |
| Precisão do capacete | 0.830 | 0.939 |
| Recall do capacete | 1.000 | 0.982 |
| mAP50 do capacete | 0.979 | 0.967 |
| mAP50 do colete | 0.978 | 0.966 |
| mAP50 geral (5 classes) | 0.927 | 0.934 |

### Metodologia, em resumo

1. Modelo base: YOLO26s já treinado no dataset epi-d4clr (`modelos/epi_backup.pt`).
2. Coleta de 239 fotos de pessoas de boné com o `coletar_bones.py`. O boné **não** recebe rótulo de capacete, então funciona como exemplo negativo.
3. Divisão: 192 fotos para treino e 47 para validação.
4. Treino com o dataset original (1272 imagens de treino) mais as fotos de boné: 1464 imagens. Validação com 276 + 47 = 323 imagens.
5. Avaliação dos dois modelos no mesmo conjunto de validação.

### Limitações observadas

- As fotos de boné vêm de poucas pessoas e poucos modelos de boné.
- O mAP50-95 do colete caiu de 0.875 para 0.826 (caixas menos justas).
- O detector associa o capacete à pessoa pela região da cabeça. Quando a pessoa é detectada cortada, o capacete pode ficar fora dessa região (ver `comparacoes/capacete_2.jpg`).

## Dataset e créditos

- Dataset base: [anderson-vlsev/epi-d4clr](https://universe.roboflow.com/anderson-vlsev/epi-d4clr), versão 12, Roboflow, licença CC BY 4.0.
- Detector de pessoas: YOLO26s com pesos COCO da Ultralytics.
- Framework: Ultralytics YOLO 8.4.
