# Relatório de treino

- Data: 26/11/2026 20:57
- Dataset: `dados_bone/data.yaml`
- Modelo base: `modelos/epi_backup.pt`
- Épocas pedidas: 50 · imgsz 640 · lote 16
- Dispositivo: NVIDIA GeForce RTX 5060 (NVIDIA/CUDA)
- Duração: 17:11

## Métricas gerais (validação)

| Métrica | Antes | Depois | Diferença |
|---|---|---|---|
| Precisão | 0.871 | 0.893 | +0.022 |
| Recall | 0.878 | 0.861 | -0.016 |
| mAP50 | 0.927 | 0.934 | +0.007 |
| mAP50-95 | 0.669 | 0.655 | -0.013 |

## Por classe

| Classe | Objetos | Precisão | Recall | mAP50 | mAP50-95 | Δ mAP50-95 |
|---|---|---|---|---|---|---|
| Balaclava | 122 | 0.972 | 0.863 | 0.963 | 0.612 | -0.023 |
| C_eletrica | 120 | 0.770 | 0.958 | 0.963 | 0.743 | -0.010 |
| Capacete (usada) | 222 | 0.939 | 0.982 | 0.967 | 0.687 | +0.001 |
| Operacional | 64 | 0.855 | 0.516 | 0.809 | 0.409 | +0.014 |
| colete (usada) | 81 | 0.928 | 0.988 | 0.966 | 0.826 | -0.048 |

## Boné confundido com capacete (fotos de validação)

- Antes: 14 de 47 fotos (30%)
- Depois: 0 de 47 fotos (0%)

## Gráficos

Na pasta do treino:

- `results.png`: curvas de loss e métricas por época
- `confusion_matrix.png`: o que o modelo confunde com o quê
- `BoxPR_curve.png`: precisão x recall por classe
- `val_batch0_pred.jpg`: exemplos de detecção na validação

## O que cada métrica significa

- **Precisão**: das caixas que o modelo desenhou, quantas estavam certas (poucos alarmes falsos).
- **Recall**: dos EPIs que existiam nas fotos, quantos o modelo achou (poucos EPIs perdidos).
- **mAP50**: qualidade geral da detecção, aceitando caixas com 50% de sobreposição.
- **mAP50-95**: igual, mas exige caixas mais justas. É a métrica mais rígida.
