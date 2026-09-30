# Revenue Risk Check

**🔗 Live now: https://live-revenue-risk-tool-api-orcin.vercel.app/**

Type in any real public company's stock ticker and instantly get a live, AI-powered risk score, pulled straight from the SEC's own filing system in real time. No cached data. No stale lookups. Every single query hits SEC's live API and runs through a trained machine learning model on the spot. ⚡

This isn't a notebook. It isn't a one-off script. It's a fully deployed, end-to-end data product: real-time data ingestion, feature engineering, and live model inference, all wired together and shipped to production. 🎯

## 🛠️ What it does

1. 🔍 You type a real ticker.
2. 📡 The backend resolves it to the company's SEC filer ID and pulls live revenue, assets, liabilities, and net income straight from data.sec.gov.
3. ⚙️ It engineers the same features the model was trained on: revenue scale, debt-to-assets, net margin, and year-over-year growth.
4. 🤖 A trained Random Forest model scores it in real time and returns a live probability of revenue decline.
5. 📊 You get a 5-year revenue history chart and a plain-language breakdown of whether each factor is running hot, cold, or right in line with typical companies.

There's also a **🏆 Leaderboard** tab that fires off live predictions for eight major companies at once and ranks them by risk, so you can see exactly how the market's biggest names stack up, right now.

## 🎯 Why the methodology is the real flex

The model hits a 0.62 ROC-AUC. Predicting a company's revenue trajectory a year out from four ratios alone is a genuinely hard problem, so this is a real, earned number, not an inflated one.

Here's the number that actually matters: a naive model that always guesses "no decline" looks like it has 78% accuracy, while catching zero real revenue declines. Zero. This model catches 50% of them. That gap is the whole story. 📈

The evaluation is also done right: a **time-based train/test split** (trained on 2009 to 2020, tested only on 2021 onward), so the model is judged the same way it would actually be used, predicting forward on data it's never seen, not shuffled randomly to look artificially strong. 🔒

## 🔥 What drives the prediction

1. 📈 Year-over-year revenue growth
2. 💰 Company size (log revenue)
3. 💵 Net profit margin
4. ⚖️ Debt-to-assets ratio

## 🏗️ Architecture

- **Frontend** (`index.html`): pure HTML, CSS, and JavaScript, zero frameworks, zero bloat.
- **Backend** (`api/predict.py`): a live Python serverless function on Vercel. Every request resolves the ticker, pulls fresh SEC data, builds the feature set, and runs real-time inference.
- **Model** (`api/model_bundle.joblib`): a Random Forest classifier trained on 505 real company-years (2009 to 2025, 48 companies), shipped with real percentile benchmarks so every result comes with instant context.

No API key required anywhere. 🔓 SEC's API is fully open, it just needs a descriptive User-Agent, which is hardcoded server-side.

## 📚 Data source

100% real. Every number comes straight from the SEC's public EDGAR API, sourced from actual 10-K filings. Ticker-to-CIK mapping comes straight from SEC's own published data too.

## ⚠️ Honest limitations

- 48 companies is a small, non-random sample of large, well-known public companies, not the whole market, so results likely wouldn't generalize to smaller companies without revalidating.
- Some companies report financials under different XBRL tags across years, creating real gaps, handled with missingness indicators rather than hidden or faked.
- The class balance shifted between training years and test years (33% revenue declines in training vs. 22% in test), reflecting real differing economic conditions, not a modeling artifact.
- This is a portfolio project, not investment advice. Nothing here should inform an actual financial decision. 🚫💸

## Running it locally
pip install pandas scikit-learn joblib requests
python pull_sec_data.py
python model.py


To run the live app itself, you'll need to deploy it (Vercel or similar), since the frontend calls a serverless backend rather than running everything in one script.

## 📁 Files

- `index.html`: the live tool's frontend
- `api/predict.py`: the serverless backend pulling live SEC data and running the model
- `api/model_bundle.joblib`: the trained model, with training medians and percentile context baked in
- `api/requirements.txt`: backend dependencies
- `pull_sec_data.py`: the original script used to pull the training dataset
- `model.py`: training and evaluation script, including the time-based split and metrics above
