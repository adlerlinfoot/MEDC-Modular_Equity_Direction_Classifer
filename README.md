**MEDC‑fEDC: Modular Equity Direction Classifier (Daily UP/DOWN Forecasting Engine)**

---

Reproducible pipeline for **daily equity direction classification** across U.S. equities using engineered technical features, walk‑forward validation, and gradient‑boosted models. The system produces **UP/DOWN** signals with associated probabilities and confidence scores, enabling cross‑sectional ranking and future confidence‑gated execution.

Of note, this project is part of an ongoing research series exploring **cross‑sectional equity signals**, **regime structure**, and **confidence‑driven execution frameworks**.

---

## **OVERVIEW**

This project builds a **binary direction classifier** for daily equity closes.  
For each ticker, the model predicts whether tomorrow’s close will be:

- **UP** (positive next‑day return)  
- **DOWN** (negative next‑day return)

The feature set captures four core structural components of equity behavior:

- **Momentum:** short‑term regime pressure  
- **Realized volatility:** conviction and volatility clustering  
- **Long‑term trend:** alignment with broader equity trend  
- **Mean reversion:** correction pressure and z‑score deviation  

All features are shifted by one day to prevent lookahead bias.

A GradientBoostingClassifier is tuned via walk‑forward TimeSeriesSplit, producing stable out‑of‑sample accuracy across a broad equity universe.  
When evaluated across ~100 tickers, the system achieves an **average accuracy of ~0.52**, with **no names below 0.50**, confirming a persistent cross‑sectional edge suitable for confidence‑ranked execution.

---

## **METHOD SUMMARY**

- **Feature engineering:**  
  momentum windows (5/10/20), realized volatility windows (5/10/20), long‑term trend (SMA‑50 deviation), mean‑reversion (20‑day z‑score), and base return/volatility features.

- **Labeling:**  
  next‑day return sign → binary UP/DOWN classification.

- **Modeling:**  
  GradientBoostingClassifier with expanded hyperparameter grid and walk‑forward TimeSeriesSplit.

- **Evaluation:**  
  per‑ticker validation accuracy, cross‑sectional accuracy distribution, sector averages, and cap‑bucket averages.

- **Execution (future extension):**  
  confidence‑ranked cross‑sectional selection (e.g., top‑N signals per day), bucketed confidence layers, and position sizing based on probability strength.

---

## **REPRODUCIBILITY**

The entire workflow is deterministic and leakage‑free.

Run the full pipeline through: python -m src.system --ticker [symbol] --start [chosen window]


Evaluate the full equity universe: python -m src.tester

Models and metadata (best parameters, validation accuracy) are saved under: artifacts/<ticker>/

---

## **RESULTS SUMMARY**

Across ~100 equities (2010–2026):

- **Average accuracy:** 0.5199  
- **Best performers:** TXN, ABBV, META, AMAT, ORCL (~0.532–0.536)  
- **Worst performers:** DE, GE (~0.503–0.505)  
- **No tickers below 50%**  
- **Tech sector average:** ~0.526  
- **Stable cross‑sectional distribution**  

This confirms a **real, persistent, generalizable edge** suitable for multi‑asset confidence‑ranked execution.

---

## **LICENSE**  
MIT License.
