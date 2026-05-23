# Meta-Aprenentatge Independent del Model per a Aprenentatge per Reforç Jeràrquic aplicat a Entorns de Trading

> Treball Final de Màster (TFM) — Universitat Oberta de Catalunya (UOC)
> **Xavier Vizcaino Gascon** · xvizcainog@uoc.edu

---

## Resum

Aquest projecte investiga si el **meta-aprenentatge independent del model (MAML / Reptile)** pot accelerar l'adaptació d'agents d'**aprenentatge per reforç jeràrquic (HRL)** a nous règims de mercat en un entorn de trading de criptomonedes.

La hipòtesi central és que un agent HRL basat en xarxes feudals (arquitectura Manager–Worker) pre-entrenat amb inicialització per meta-aprenentatge s'adaptarà més ràpidament i obtindrà un millor rendiment ajustat al risc que un agent PPO estàndard en dades no vistes.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│               Entorn de Trading                     │
│   BTC/USD · 60 min · espai d'observació 71 dims     │
│   CryptoMarketEnv (dense) /                         │
│   CryptoMarketEnvConsolidatedReward (sparse)        │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │   FeudalAgentV5       │
          │   46.786 paràmetres   │
          │                       │
          │  PerceptionModule     │
          │  71 → 128 → 64        │
          │                       │
          │  ManagerModule (LSTM) │
          │  h = 32               │
          │                       │
          │  WorkerModule (LSTM)  │
          │  h = 32               │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  Capa de Meta-        │
          │  Aprenentatge         │
          │  FOMAML / Reptile     │
          │  _adj | _new init     │
          │  Fixed | Loss budget  │
          └───────────────────────┘
```

**Agent base PPO** — 35.269 paràmetres; MLP embut [128, 64], activació tanh, backbones actor-crític independents.

---

## Estructura del Repositori

```
repo/
├── notebooks/
│   ├── NB01_gestio_de_dades.ipynb          # Adquisició de dades i enginyeria de característiques
│   ├── dense/
│   │   ├── NB02_entorn.ipynb               # Entorn de trading amb recompensa densa
│   │   ├── NB03_agent_base.ipynb           # Agent base PPO (dense)
│   │   ├── NB04_hrl_v0..v9.ipynb           # Agents HRL v0–v9 (dense)
│   │   ├── NB05_reptile.ipynb              # Meta-aprenentatge Reptile (dense)
│   │   ├── NB06_fomaml.ipynb               # Meta-aprenentatge FOMAML (dense)
│   │   └── NB07_avaluacio.ipynb            # Avaluació — 4 experiments (dense)
│   └── sparse/
│       ├── NB02_entorn.ipynb               # Entorn de trading amb recompensa dispersa
│       ├── NB03_agent_base.ipynb           # Agent base PPO (sparse)
│       ├── NB04_hrl_v0..v6.ipynb           # Agents HRL v0–v6 (sparse)
│       ├── NB05_reptile.ipynb              # Meta-aprenentatge Reptile (sparse)
│       ├── NB06_fomaml.ipynb               # Meta-aprenentatge FOMAML (sparse)
│       └── NB07_avaluacio.ipynb            # Avaluació — 4 experiments (sparse)
├── resultats/
│   ├── Experiments/
│   │   ├── Dense/                          # Gràfics de resultats (PNG)
│   │   └── Sparse/                         # Gràfics de resultats (PNG)
│   └── Hiperparametres/                    # Resultats de la cerca d'hiperparàmetres
└── src/
    ├── common/                             # Utilitats, mètriques, visualització
    ├── connectors/                         # Connector API de Kraken
    ├── data/                               # Càrrega i preprocessament de dades
    ├── features/                           # Mòduls d'indicadors tècnics
    │   ├── momentum.py
    │   ├── price_action.py
    │   ├── trend.py
    │   ├── volatility.py
    │   └── volume.py
    └── rl/
        ├── agents/                         # FeudalAgent v0–v9 + base
        ├── envs/                           # Entorns compatibles amb Gymnasium
        ├── meta_learning/                  # Implementacions FOMAML i Reptile
        ├── models/                         # MLP, policy heads, value heads
        └── training/                       # Bucles d'entrenament i utilitats
```

---

## Dades

| Propietat | Valor |
|---|---|
| Actiu | BTC/USD |
| Font | API de Kraken |
| Granularitat | 60 minuts |
| Període | 2018 – 2025 |
| Conjunt d'entrenament | 2018 – 2023 |
| Conjunt de validació | 2023 |
| Conjunt de test | 2024 – 2025 |

**Enginyeria de característiques**: 25 indicadors tècnics (momentum, price action, tendència, volatilitat, volum) → 65 característiques normalitzades `Norm_*` + 6 variables d'estat intern = **espai d'observació de 71 dimensions**.

---

## Experiments d'Avaluació

S'apliquen quatre protocols d'avaluació a cada variant d'agent:

| # | Nom | Finestra | Dimensionament |
|---|---|---|---|
| Ex.1 | Rolling all-in | Rolling | Tot el capital |
| Ex.2 | Rolling sizing | Rolling | Dimensionament proporcional |
| Ex.3 | Full period all-in | Període complet | Tot el capital |
| Ex.4 | Full period sizing | Període complet | Dimensionament proporcional |

---

## Variants de Meta-Aprenentatge

Cada algorisme de meta-aprenentatge s'avalua amb dues estratègies d'inicialització i dos pressupostos d'adaptació:

- **`_adj`** — inicialització càlida des d'un checkpoint PPO pre-entrenat
- **`_new`** — inicialització aleatòria

- **`Fixed`** — pressupost d'adaptació fix (1.000 passos)
- **`Loss`** — pressupost d'adaptació basat en convergència

---

## Posada en Marxa

### Requisits

```bash
pip install torch gymnasium polars pandas numpy matplotlib seaborn
pip install krakenex codecarbon
```

### Seqüència de notebooks

Executar els notebooks en ordre dins de cada variant de recompensa (`dense/` o `sparse/`):

1. **NB01** — Descàrrega de dades OHLCV de Kraken i càlcul d'indicadors tècnics
2. **NB02** — Definició i validació de l'entorn de trading
3. **NB03** — Entrenament i avaluació de l'agent PPO base
4. **NB04** — Entrenament de variants de l'agent HRL (v0 → v9 per a dense, v0 → v6 per a sparse)
5. **NB05** — Aplicació de Reptile sobre el millor agent HRL
6. **NB06** — Aplicació de FOMAML sobre el millor agent HRL
7. **NB07** — Execució dels 4 experiments d'avaluació i generació de gràfics

---

## Infraestructura d'Entrenament

- **Aprenentatge per currículum**: planificació del `fee_factor` (0 → 1 al llarg dels passos d'entrenament) per introduir gradualment els costos de transacció
- **Seguiment energètic**: integració de CodeCarbon per monitorar l'empremta de carboni en tots els entrenaments
- **Avaluació walk-forward**: protocol de finestra mòbil per evitar biaixos de mirada enrere

---

## Llicència

Vegeu [LICENSE](LICENSE) per als detalls.
