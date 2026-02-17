# Signal Processing and Optimization for Big Data - Progetto d' Essame - Università degli Studi di Perugia

# Model-Based Deep Learning per RPCA nella Decomposizione del Traffico Internet
### *From Pure Model-Based Optimization to Adaptive Model-Based Deep Learning*

> Progetto per il corso **Signal Processing and Optimization for Big Data**  
> Corso di Laurea Magistrale in Ingegneria Informatica e Robotica — Data Science and Data Engineering  
> Università degli Studi di Perugia  
> *Daniele Angeloni*

---

## Descrizione

Questo progetto studia il problema della **decomposizione di matrici di traffico Internet** secondo il modello strutturale *Low-Rank plus Sparse* (RPCA):

$$Y = L_0 + S_0 + N$$

dove $L_0$ rappresenta il traffico nominale (componente a basso rango), $S_0$ le anomalie strutturate (componente sparsa) e $N$ il rumore additivo.

L'obiettivo non è la semplice anomaly detection, ma la **ricostruzione accurata di entrambe le componenti latenti**, analizzando in modo sistematico il passaggio da un approccio puramente model-based — basato sull'algoritmo ADMM applicato al problema convesso di Principal Component Pursuit — a formulazioni progressive di **Model-Based Deep Learning** tramite deep unfolding e operatori learnable.

Il punto centrale del lavoro è lo studio del **model mismatch**: quando le ipotesi teoriche di RPCA non sono pienamente soddisfatte (sottospazi tempo-varianti, anomalie strutturate e correlate, rumore elevato), l'ADMM classico degrada significativamente. L'integrazione controllata di componenti apprendibili all'interno della struttura algoritmica consente di compensare questo mismatch mantenendo interpretabilità e coerenza con la formulazione originale.

---

## Approcci Implementati

La progressione metodologica è strutturata in sei configurazioni sperimentali:

| Run | Nome | Descrizione |
|-----|------|-------------|
| RUN1 | ADMM Ideale | ADMM classico su dataset ideale (assunzioni teoriche soddisfatte) |
| RUN2 | ADMM Mismatch (Baseline) | Stesso ADMM applicato al dataset con mismatch strutturale |
| RUN3 | Global-Learned ADMM | Selezione data-driven globale degli iperparametri λ e ρ tramite GridSearch |
| RUN4 | Layer-wise Unfolded ADMM | Deep unfolding con parametri λ_k e ρ_k learnable per layer |
| RUN5 | Unfolded ADMM + Learned Prox | Sostituzione del soft-thresholding con un operatore convoluzionale learnable |
| RUN6 | Hybrid Residual Unfolded ADMM | Correzioni residue learnable su entrambe le componenti (L e S) |

---

## Struttura della Repository

```
├── Dataset_Generation/
│   ├── generate_dataset_ideal.m          # Generazione dataset ideale (matched assumptions)
│   └── generate_dataset_mismatch.m       # Generazione dataset con mismatch strutturale
│
├── Results_Figures_Utils/
│   ├── results_figures/                  # Figure e grafici dei risultati sperimentali
│   ├── results_metrics/                  # Metriche numeriche per ogni RUN
│   └── utils/                            # Funzioni di supporto per visualizzazione e analisi
│
├── old_MatlabVersionCode/                # Versione prototipale in MATLAB (riferimento storico)
│   ├── figures/
│   ├── utils/
│   ├── generate_dataset_ideal.m
│   ├── generate_dataset_mismatch.m
│   ├── run1_admm_ideal.m
│   ├── run2_admm_mismatch.m
│   └── run3_Learned_ADMM_mismatch.m
│
├── projectNotebook.ipynb                 # Notebook principale con tutti i RUN (Python)
├── projectNotebook.html                  # Versione HTML del notebook (visualizzazione statica)
├── projectNotebook.pdf                   # Versione PDF del notebook
└── README.md
```

---

## Metriche di Valutazione

Le prestazioni vengono valutate tramite:

- **NMSE_L** — Normalized Mean Squared Error sulla componente low-rank  
- **NMSE_S** — Normalized Mean Squared Error sulla componente sparsa  

Con analisi secondaria su convergenza, efficienza iterativa e stabilità in regime mismatch.

### Risultati Sintetici

| Run | NMSE_L | NMSE_S | Iter / Layer |
|-----|--------|--------|--------------|
| RUN1 (Ideale) | 0.00058 | 0.0235 | 433 |
| RUN2 (Mismatch Baseline) | 0.0538 | 0.822 | 188 |
| RUN3 (Global Learned) | 0.0538 | 0.822 | 95 |
| RUN4 (Unfolded Layer-wise) | 0.0521 | 0.425 | 10 |
| RUN5 (Unfolded + Learned Prox) | 0.0356 | 0.282 | 10 |
| RUN6 (Hybrid Residual) | **0.0191** | **0.121** | 10 |

---

## Requisiti

Il notebook principale è sviluppato in **Python**. Le dipendenze principali sono:

```
numpy
scipy
matplotlib
torch
```

La generazione dei dataset è disponibile anche in versione **MATLAB** nella cartella `old_MatlabVersionCode/`.

---

## Riferimenti

- Candès, Li, Ma, Wright — *Robust Principal Component Analysis?*, JACM 2011  
- Boyd et al. — *Distributed Optimization via ADMM*, 2011  
- Shlezinger, Eldar, Boyd — *Model-Based Deep Learning*, 2022  
- Monga, Li, Eldar — *Algorithm Unrolling*, IEEE SPM 2021  
- Solomon et al. — *Deep Unfolded Robust PCA*, IEEE 2021
