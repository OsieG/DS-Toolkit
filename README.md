## DS Toolkit

**[Try the live app →](https://ds-toolkit-5nwafpfkq9zsmcz2ryphzh.streamlit.app/)**

A small end-to-end data science toolkit built around the Kaggle API. Search for a dataset, download it, explore it, clean it, and train a baseline model, all from one interface.

Built as a hands-on project to practice the full workflow a data science role actually touches: API design, data cleaning, preprocessing, model training, and shipping a usable frontend on top of it.

## What it does

- **Search and download** any Kaggle dataset by keyword, with local caching so repeat downloads don't hit the Kaggle API again
- **Explore** a dataset's shape, column types, missing values, and summary statistics
- **Preprocess** with configurable imputation (mean/median/mode/drop), encoding (one-hot/label), and scaling (standard/min-max), with a before/after preview
- **Check data quality**: exact-match duplicate detection (with optional column subsetting to exclude ID-like columns) and IQR-based outlier detection
- **Inspect correlations** between numeric columns
- **Split** into train/validation/test sets with configurable ratios
- **Train a baseline model** (logistic regression for classification, linear regression for regression) and see accuracy/precision/recall/F1 or MSE/R², a confusion matrix or predicted-vs-actual chart, and feature coefficients
- **Export** the processed dataset as a CSV

## Architecture

```
backend/    FastAPI service — all the actual data processing
frontend/   Streamlit app — calls the backend over HTTP, renders results
```

The backend does all the work; the frontend is a thin client that calls it and displays what comes back. Each lives in its own Python virtual environment with its own `requirements.txt`.

## Tech stack

- **Backend:** FastAPI, pandas, scikit-learn, the `kaggle` API client
- **Frontend:** Streamlit, matplotlib
- **Data source:** [Kaggle](https://www.kaggle.com)

## Running it locally

### Backend

1. Get a Kaggle API token from your [Kaggle account settings](https://www.kaggle.com/settings/account) and place it at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY` as environment variables instead)
2. ```bash
   cd backend
   python -m venv env
   env\Scripts\activate      # Windows
   # source env/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```
3. Visit `http://localhost:8000/docs` to confirm it's running

### Frontend

1. ```bash
   cd frontend
   python -m venv env
   env\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
2. Make sure `BACKEND_URL` in `app.py` points at wherever the backend is running

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/search` | GET | Search Kaggle datasets by keyword |
| `/download` | POST | Download (or reuse cached) dataset files |
| `/stats` | GET | Shape, dtypes, null counts, summary statistics |
| `/preprocess` | GET | Imputation, encoding, scaling with before/after preview |
| `/correlation_matrix` | GET | Correlation between numeric columns |
| `/duplicates` | GET | Duplicate row detection, with optional removal |
| `/outliers` | GET | IQR-based outlier detection per column |
| `/split` | GET | Train/validation/test split |
| `/train` | GET | Train a baseline classification or regression model |
| `/export` | GET | Download the processed dataset as a CSV |

## Known limitations

- The numeric-vs-categorical column heuristic (used to decide which columns to offer for encoding vs. scaling in the UI) is based on whether a column has a numeric mean, so a binary numeric column like a `0`/`1` survival flag is treated as numeric rather than categorical. Doesn't affect the backend's correctness, just which columns the frontend suggests by default.
- No session isolation: the backend uses a single shared file cache per dataset, so concurrent users would share cached files. Fine for a single-user demo, not built for multi-user production use.
- Hosted on free tiers, so the backend may take 30–60 seconds to respond after a period of inactivity while it wakes back up.
