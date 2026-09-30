"""
Vercel Python serverless function.
Given a stock ticker, pulls that company's LIVE financial data straight from
SEC's public EDGAR API, computes the same features the model was trained on,
and returns a real-time revenue-decline risk prediction.

No API key needed anywhere in this function — SEC's API is fully open,
it only requires a descriptive User-Agent, which is not a secret.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import numpy as np
import pandas as pd
import joblib

HEADERS = {"User-Agent": "Bhakti Pasnani Portfolio Project bhaktipasnani02@gmail.com"}
CONCEPTS = ["Revenues", "Assets", "Liabilities", "NetIncomeLoss"]
REVENUE_FALLBACK = "RevenueFromContractWithCustomerExcludingAssessedTax"

_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "model_bundle.joblib")
_bundle = joblib.load(_BUNDLE_PATH)
MODEL = _bundle["model"]
MEDIANS = _bundle["medians"]
FEATURES = _bundle["features"]

_cik_cache = {"map": None}


def _fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def get_cik(ticker):
    if _cik_cache["map"] is None:
        data = _fetch_json("https://www.sec.gov/files/company_tickers.json")
        _cik_cache["map"] = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}
    return _cik_cache["map"].get(ticker.upper())


def get_concept_facts(cik, concept):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
    try:
        return _fetch_json(url)
    except Exception:
        return None


def extract_annual_values(facts_json):
    if not facts_json:
        return {}
    annual = {}
    try:
        for entry in facts_json.get("units", {}).get("USD", []):
            if entry.get("form") == "10-K" and entry.get("fp") == "FY":
                fy, val = entry.get("fy"), entry.get("val")
                if fy and val is not None:
                    annual[fy] = val
    except Exception:
        pass
    return annual


def build_features_for_company(ticker):
    cik = get_cik(ticker)
    if not cik:
        return None, f"No SEC CIK found for ticker '{ticker}'. Try the exact exchange ticker (e.g. AAPL, MSFT)."

    data = {}
    for concept in CONCEPTS:
        facts = get_concept_facts(cik, concept)
        if concept == "Revenues" and not facts:
            facts = get_concept_facts(cik, REVENUE_FALLBACK)
        data[concept] = extract_annual_values(facts)

    revenues = data.get("Revenues", {})
    if not revenues:
        return None, f"No revenue data found in SEC filings for '{ticker}'."

    years = sorted(revenues.keys())
    latest_fy = years[-1]
    prev_fy = years[-2] if len(years) > 1 else None

    rev_latest = revenues.get(latest_fy)
    rev_prev = revenues.get(prev_fy) if prev_fy else None
    assets = data.get("Assets", {}).get(latest_fy)
    liabilities = data.get("Liabilities", {}).get(latest_fy)
    net_income = data.get("NetIncomeLoss", {}).get(latest_fy)

    debt_to_assets = (liabilities / assets) if (assets and liabilities) else None
    net_margin = (net_income / rev_latest) if (net_income is not None and rev_latest) else None
    revenue_growth_yoy = ((rev_latest - rev_prev) / rev_prev) if (rev_prev and prev_fy == latest_fy - 1) else None

    feature_row = {
        "log_revenue": np.log(rev_latest) if rev_latest else None,
        "debt_to_assets": debt_to_assets if debt_to_assets is not None else MEDIANS["debt_to_assets"],
        "net_margin": net_margin if net_margin is not None else MEDIANS["net_margin"],
        "revenue_growth_yoy": revenue_growth_yoy if revenue_growth_yoy is not None else MEDIANS["revenue_growth_yoy"],
        "debt_to_assets_missing": int(debt_to_assets is None),
        "net_margin_missing": int(net_margin is None),
        "growth_missing": int(revenue_growth_yoy is None),
    }

    raw_values = {
        "fiscal_year": latest_fy,
        "revenue": rev_latest,
        "assets": assets,
        "liabilities": liabilities,
        "net_income": net_income,
        "debt_to_assets": debt_to_assets,
        "net_margin": net_margin,
        "revenue_growth_yoy": revenue_growth_yoy,
    }

    return {"features": feature_row, "raw": raw_values}, None


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            ticker = (body.get("ticker") or "").strip().upper()

            if not ticker:
                self._send(400, {"error": "Missing ticker"})
                return

            result, error = build_features_for_company(ticker)
            if error:
                self._send(404, {"error": error})
                return

            X = pd.DataFrame([result["features"]])[FEATURES]
            proba = MODEL.predict_proba(X)[0][1]
            prediction = "high_risk" if proba >= 0.5 else "low_risk"

            self._send(200, {
                "ticker": ticker,
                "prediction": prediction,
                "risk_probability": round(float(proba), 4),
                "raw_financials": result["raw"],
            })
        except Exception as e:
            self._send(500, {"error": str(e)})

    def _send(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
