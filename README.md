# Revenue Risk Check

**Live at: https://live-revenue-risk-tool-api-orcin.vercel.app/**

Type in any real public company's stock ticker and it pulls that company's actual latest financial filings straight from the SEC, right then, and runs them through a model trained on real historical filing data to predict whether the company's revenue is likely to decline next fiscal year.

This isn't a notebook or a one time script. It's a small, real, deployed system. Every time someone types a ticker, it hits SEC's live API, not a cached lookup.

## Why I built this

I wanted a data science project that used real data I pulled myself, not a cleaned dataset someone else packaged for a tutorial. So I wrote a script that pulls actual 10-K filing data directly from the SEC's public EDGAR API, revenue, assets, liabilities, and net income, for about 50 public companies going back to 2009. Then I built a model predicting whether a company's revenue would decline the next year, using only that year's financials.

I also wanted it to feel like something real, not just a printout of numbers. So instead of stopping at a script, I turned it into a live tool anyone can actually use.

## What it does

1. You type a real ticker.
2. The backend resolves it to the company's SEC filer ID and pulls its live revenue, assets, liabilities, and net income straight from data.sec.gov.
3. It computes the same features the model was trained on (revenue size, debt to assets, net margin, year over year revenue growth).
4. The trained model scores it and returns a probability of revenue decline next year.
5. You also get a five year revenue history chart and a plain explanation of whether each factor is typical, above typical, or below typical compared to the training data.

There's also a Leaderboard tab that pulls live scores for eight well known companies at once and sorts them by risk, so you can see how a handful of major companies stack up right now.

## Why the methodology matters more than the headline number

The model's ROC AUC is 0.62. That is honestly modest, and I think that's the right outcome to be upfront about. Predicting a specific company's revenue trajectory a year out from four basic ratios alone is a genuinely hard problem. If it were easy, that edge wouldn't exist, markets would already price it in.

What actually matters here is the comparison. A naive model that always predicts "no decline" would look like it has 78 percent accuracy, while catching exactly zero real revenue declines. This model catches 50 percent of actual declines. That gap between a misleading accuracy number and what the model actually catches is the real point of this project, not the AUC number by itself.

I also used a time based train and test split on purpose. The model trained on 2009 through 2020 and was tested only on 2021 and later. Companies repeat across years in this dataset, so a random split would let future years leak into training and make the results look better than they honestly are. A time based split mimics how this would actually be used: trained on history, predicting forward on new filings.

## What drives the prediction

In order of importance:
1. Year over year revenue growth
2. Company size (log revenue)
3. Net profit margin
4. Debt to assets ratio

## Architecture

- **Frontend** (`index.html`): plain HTML, CSS, and JavaScript. No framework. Handles the ticker lookup, the leaderboard, and rendering results.
- **Backend** (`api/predict.py`): a Python serverless function on Vercel. On every request it resolves the ticker's CIK, pulls live data from SEC, builds the feature row, and runs the trained model.
- **Model** (`api/model_bundle.joblib`): a Random Forest classifier trained on 505 real company years of filing data (2009 to 2025, 48 companies), saved with the real percentile distributions used to flag whether a company's numbers are typical or not.

No API key is needed anywhere. SEC's API is fully open. It only requires a descriptive User Agent identifying who is making the request, which is not a secret and is hardcoded in the backend.

## Data source

All financial data comes from the SEC's public EDGAR API (data.sec.gov), pulled directly from real 10-K filings. Company ticker to CIK mapping also comes straight from SEC's own published list.

## Honest limitations

- 48 companies is a small, non random sample. These are large, well known public companies, not a representative slice of the market, and results likely wouldn't generalize to smaller companies without revalidating.
- Some companies report financials under different XBRL tags across years, which creates real gaps in the data. Those gaps are handled with missingness indicators rather than hidden or faked.
- The class balance shifted between the training years and the test years (33 percent revenue declines in training years versus 22 percent in test years), which reflects real differing economic conditions between the two periods, not a modeling artifact, but it's worth knowing about.
- This is a portfolio project, not investment advice. Nothing here should inform an actual financial decision.

## Running it locally
