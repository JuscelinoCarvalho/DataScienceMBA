# Reanalysis / Reanálise (v2)

**EN:** Critical reanalysis of the original thesis (v1). The full story, with tables and figures, is in the
[main README (English)](../README.en.md#-retrospective-what-was-wrong-in-v1).
**PT:** Reanálise crítica do TCC original (v1). A explicação completa, com tabelas e figuras, está no
[README principal (português)](../README.md#-retrospectiva-o-que-estava-errado-na-v1).

**AI transparency:** this reanalysis was carried out with AI assistance (Claude) under my direction; all results
were re-run and verified.
**Transparência sobre IA:** esta reanálise foi feita com apoio de IA (Claude), sob minha direção. Todos os
resultados foram reexecutados e verificados.

| File / Arquivo | EN | PT |
|---|---|---|
| `01_diagnose_v1.py` | Reproduces the v1 pipeline and changes **one factor at a time** (leaky feature, negative cut-off) to measure each flaw | Reproduz o pipeline da v1 e muda **um fator por vez** (feature com vazamento, corte dos negativos) para medir cada falha |
| `02_corrected_pipeline_v2.py` | Corrected pipeline: no leakage, matched cut-offs, same inclusion rule for both classes, OOF threshold, bootstrap CIs, repeated CV and temporal validation | Pipeline corrigido: sem vazamento, cortes pareados, mesma regra de inclusão para as duas classes, threshold OOF, IC por *bootstrap*, CV repetida e validação temporal |
| `03_make_figures.py` | Builds the README figures from the JSON results (no data needed) | Gera as figuras do README a partir dos JSONs (não precisa dos dados) |
| `results/results_v1_diagnosis.json` | v1 ablation metrics | Métricas da ablação da v1 |
| `results/results_v2.json` | v2 metrics, hyperparameters and top features | Métricas, hiperparâmetros e principais variáveis da v2 |
| `figures/` | Summary figures and v2 precision–recall curves | Figuras-resumo e curvas Precisão × Recall da v2 |

Run from the repository root with the Kaggle data in `./Data/training_data/`. The generated tabular datasets
(`reanalysis/bases/`) are not versioned, because the Kaggle data cannot be redistributed.
Rode a partir da raiz do repositório, com os dados do Kaggle em `./Data/training_data/`. As bases tabulares
geradas (`reanalysis/bases/`) não são versionadas, porque os dados do Kaggle não podem ser redistribuídos.
