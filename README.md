# AI-Augmented Real Estate Market Intelligence Platform

End-to-end analytics platform over **CRMLS California residential** MLS data
(28 months, ~350K closed sales). It ingests data from live APIs, runs a
reproducible cleaning/feature pipeline, and layers three AI modules on top —
an LLM market-report writer, a natural-language data assistant, and an
explainable home-valuation model — plus interactive Tableau dashboards.

> Built during a data-analyst internship and extended into an **AI-augmented**
> analytics project: AI is used as a force-multiplier for analysis and
> reporting (auto-narration, self-serve querying, decision-support modeling),
> with deterministic code doing the math and LLMs doing the language.

**Confidentiality:** the underlying MLS data is confidential and is **not**
included in this repo (CSVs, `.twbx`, SQL extracts, and the API extraction
scripts are gitignored). All AI calls can run **fully locally** (Ollama), so
confidential data never leaves the machine.

---

## Architecture

```mermaid
flowchart TD
    A[CoreLogic Trestle API<br/>OData REST] -->|crmls_sold/listed.py| B[Monthly CSVs]
    F[FRED API<br/>30yr mortgage rate] -->|connectors/fred_connector.py| C
    B --> C[Data pipeline<br/>week1–8 + run_pipeline.py]
    C --> D[(Cleaned, feature-engineered<br/>Sold & Listed datasets)]
    D --> E[Tableau dashboards<br/>market_analysis / competitive_analysis]
    D --> G[M1 · LLM market report<br/>ai/reporting]
    D --> H[M2 · Chat-with-data<br/>ai/assistant + app.py]
    D --> I[M3 · AVM + SHAP<br/>ai/models/avm]
    D --> K[M4 · Forecasting + alerts<br/>ai/forecast]
    G & H & I --> J[ai/shared/llm.py<br/>provider-agnostic LLM client]
```

---

## AI modules

### M1 · LLM market-report generator  ·  `ai/reporting/market_narrative.py`
Replaces hand-written market commentary. Computes monthly KPIs
(MoM/YoY/peak/supply-demand), feeds the LLM a **pre-labelled fact sheet**
(so small models don't misread MoM vs YoY or signs), and gets back a structured
`{headline, summary, watch[]}` narrative that is embedded into the EDA report.
Falls back to a deterministic stub if no LLM is available.

```bash
.venv/bin/python eda_report.py          # report with an auto "AI 市场综述" section
```

### M2 · Chat-with-data (text-to-SQL)  ·  `ai/assistant/text_to_sql.py` + `app.py`
Lets non-technical stakeholders query the data in plain language. The LLM writes
**DuckDB SQL** against the sold table; a safety layer allows only a single
`SELECT` (DDL/DML rejected) and wraps every query in an outer `LIMIT`. A
Streamlit UI shows the generated SQL, a result table, an auto chart, and a CSV
export.

```bash
.venv/bin/streamlit run app.py
```

### M3 · AVM home-valuation model + SHAP  ·  `ai/models/avm/train_avm.py`
An explainable pricing decision-support model. XGBoost predicts `ClosePrice`
from **leakage-free** features (size, year, lot, garage, lat/lon, county,
sub-type, amenities, mortgage rate) — deliberately excluding `ListPrice`,
`price_per_sqft`, `DaysOnMarket`, etc. SHAP surfaces the top value drivers.

```bash
.venv/bin/python ai/models/avm/train_avm.py    # writes metrics + SHAP plot to outputs/
```
Test performance: **R² ≈ 0.875, median APE ≈ 8.1%, 58% of predictions within
±10%**. Top SHAP drivers: **longitude, latitude, living area** — i.e. location
and size. (Good-but-not-perfect metrics confirm the model is leakage-free.)

### M4 · Market forecasting + alerting  ·  `ai/forecast/forecast_market.py`
Forecasts monthly **median close price** and **closed sales** 3 months ahead
with SARIMAX (80% confidence intervals), backtests accuracy on a holdout, and
raises a deviation alert when the latest month falls outside expectation. Uses a
seasonal-AR model so the intervals stay realistic on the short (~28-month)
series. Backtest MAPE: price ≈ 2.4%, sales ≈ 12.7%.

```bash
.venv/bin/python ai/forecast/forecast_market.py   # writes forecast.png + forecast.json to outputs/
```

M1–M3 share one **provider-agnostic LLM client** (`ai/shared/llm.py`): switch
between a local model and a cloud API with one env var, no code change.

---

## Data pipeline

A 12-stage pipeline keeps the Sold and Listed chains in sync; run it whenever a
new month of raw data is added:

```bash
.venv/bin/python run_pipeline.py            # full pipeline
.venv/bin/python run_pipeline.py --from week6   # resume from a stage
.venv/bin/python run_pipeline.py --list     # list stages
```

Stages: monthly concat + Residential filter → EDA/validation + **FRED mortgage
rate enrichment** → cleaning + date/geo checks → feature engineering → IQR
outlier flagging → Tableau prep → EDA report.

---

## Tech stack

`Python` · `pandas` / `numpy` · `DuckDB` · `XGBoost` · `SHAP` · `scikit-learn`
· `Streamlit` · `Anthropic Claude` / `Ollama (local LLM)` · `matplotlib` ·
`Tableau Public` · live REST APIs (`CoreLogic Trestle`, `FRED`)

---

## Setup

```bash
# 1. Python env (use the venv's python, not a system/brew python)
/usr/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. (Optional) free local LLM for M1/M2
brew install ollama libomp        # libomp is also required by xgboost on macOS
brew services start ollama
ollama pull qwen2.5:3b

# 3. Config — copy and fill in
cp .env.example .env              # set LLM_PROVIDER, optional API keys
```

`.env` keys: `LLM_PROVIDER` (`ollama` | `anthropic` | `stub`), `OLLAMA_MODEL`,
`ANTHROPIC_API_KEY`, `FRED_API_KEY`. `.env` is gitignored — never commit secrets.

---

## Project structure

```
.
├── connectors/fred_connector.py     # FRED REST API connector (key auth, JSON)
├── ai/
│   ├── shared/llm.py                # provider-agnostic LLM client
│   ├── reporting/market_narrative.py# M1 — LLM market report
│   ├── assistant/text_to_sql.py     # M2 — NL → DuckDB SQL
│   ├── models/avm/train_avm.py      # M3 — AVM + SHAP
│   └── forecast/forecast_market.py  # M4 — SARIMAX forecasting + alerts
├── app.py                           # M2 — Streamlit chat-with-data UI
├── eda_report.py                    # self-contained HTML EDA report (embeds M1)
├── run_pipeline.py                  # pipeline orchestrator
├── week{1..8}_*.py                  # pipeline stages (Sold & Listed chains)
├── requirements.txt · .env.example · AI_ROADMAP.md
└── (gitignored) data/  *.csv  *.twbx  *.sql  crmls_*.py  .env  .venv  outputs/
```

---

## Skills demonstrated

Live API integration (REST, OAuth-style token auth, JSON, pagination) ·
reproducible data pipelines · feature engineering · data-quality / outlier
handling · **LLM application development** (prompt engineering, structured
output, local + cloud backends, graceful degradation) · **text-to-SQL** ·
**explainable ML** (XGBoost + SHAP, leakage control) · **time-series
forecasting** (SARIMAX, backtesting) · BI dashboards (Tableau) · secrets hygiene.

> See [`AI_ROADMAP.md`](AI_ROADMAP.md) for the module plan and remaining work.
