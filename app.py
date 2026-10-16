import os
import pickle
import warnings
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request, Response

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Order MUST match the order used during training
FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]

CURRENT_YEAR = datetime.now().year

# (min, max) allowed for each input
LIMITS = {
    "Square_Footage": (100, 20000),
    "Num_Bedrooms": (0, 15),
    "Num_Bathrooms": (0, 15),
    "Year_Built": (1800, CURRENT_YEAR + 2),
    "Lot_Size": (0, 100),
    "Garage_Size": (0, 10),
    "Neighborhood_Quality": (1, 10),
}

app = Flask(__name__)


def money(x):
    return "${:,.0f}".format(x)


def quality_label(q):
    if q >= 9:
        return "an excellent"
    if q >= 7:
        return "a good"
    if q >= 5:
        return "an average"
    if q >= 3:
        return "a below-average"
    return "a poor"


def explain(v, price):
    sqft = int(v["Square_Footage"])
    beds = int(v["Num_Bedrooms"])
    baths = int(v["Num_Bathrooms"])
    year = int(v["Year_Built"])
    garage = int(v["Garage_Size"])
    quality = int(v["Neighborhood_Quality"])

    age = max(CURRENT_YEAR - year, 0)
    age_txt = "brand new" if age <= 1 else f"about {age} years old"
    garage_txt = (
        "no garage" if garage == 0
        else f"a garage for {garage} car{'s' if garage > 1 else ''}"
    )

    summary = (
        f"A {sqft:,} sq ft home with {beds} bedroom{'s' if beds != 1 else ''} and "
        f"{baths} bathroom{'s' if baths != 1 else ''}, {age_txt} (built in {year}), "
        f"with {garage_txt} in {quality_label(quality)} neighbourhood "
        f"(rated {quality}/10) is estimated to be worth around {money(price)}."
    )
    return summary


@app.route("/")
def home():
    return Response(HTML, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    values = {}

    for name in FEATURES:
        raw = data.get(name)
        if raw in (None, ""):
            return jsonify(error=f"Please enter a value for {name.replace('_', ' ')}."), 400
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return jsonify(error=f"{name.replace('_', ' ')} must be a number."), 400
        lo, hi = LIMITS[name]
        if not (lo <= val <= hi):
            return jsonify(
                error=f"{name.replace('_', ' ')} must be between {lo} and {hi}."
            ), 400
        values[name] = val

    X = np.array([[values[n] for n in FEATURES]], dtype=float)
    price = float(model.predict(X)[0])

    if price <= 0:
        return jsonify(
            error="These inputs fall outside the range the model was trained on, "
                  "so a realistic price could not be estimated. Please adjust the values."
        ), 422

    low, high = price * 0.90, price * 1.10
    return jsonify(
        price=money(price),
        low=money(low),
        high=money(high),
        per_sqft=money(price / values["Square_Footage"]),
        summary=explain(values, price),
    )


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>House Price Predictor</title>
<script>
  (function () {
    var t = null;
    try { t = localStorage.getItem("theme"); } catch (e) {}
    if (!t) t = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
  })();
</script>
<style>
  :root {
    --bg: #f4f6fb; --card: #ffffff; --text: #111827; --muted: #6b7280;
    --border: #e5e7eb; --input: #f9fafb; --accent: #4f46e5; --accent2: #7c3aed;
    --ok-bg: #eef2ff; --err-bg: #fef2f2; --err: #b91c1c; --shadow: 0 10px 30px rgba(17,24,39,.08);
  }
  [data-theme="dark"] {
    --bg: #0b1020; --card: #151b2e; --text: #e5e7eb; --muted: #9ca3af;
    --border: #263049; --input: #0f1526; --accent: #818cf8; --accent2: #a78bfa;
    --ok-bg: #1b2142; --err-bg: #2a1417; --err: #fca5a5; --shadow: 0 10px 30px rgba(0,0,0,.4);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    transition: background .25s, color .25s;
  }
  .wrap { max-width: 920px; margin: 0 auto; padding: 28px 18px 48px; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 26px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
    background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; font-size: 22px;
  }
  h1 { font-size: 1.25rem; margin: 0; }
  .sub { color: var(--muted); font-size: .85rem; margin-top: 2px; }
  .toggle {
    border: 1px solid var(--border); background: var(--card); color: var(--text);
    border-radius: 999px; padding: 8px 14px; cursor: pointer; font-size: .9rem;
  }
  .grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 20px; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 18px;
    padding: 22px; box-shadow: var(--shadow);
  }
  .card h2 { margin: 0 0 4px; font-size: 1.05rem; }
  .card p.hint { margin: 0 0 18px; color: var(--muted); font-size: .85rem; }
  .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 480px) { .fields { grid-template-columns: 1fr; } }
  label { display: block; font-size: .8rem; font-weight: 600; margin-bottom: 6px; }
  label small { color: var(--muted); font-weight: 400; }
  input {
    width: 100%; padding: 11px 12px; border-radius: 10px; border: 1px solid var(--border);
    background: var(--input); color: var(--text); font-size: .95rem; outline: none;
    transition: border-color .15s, box-shadow .15s;
  }
  input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent); }
  .full { grid-column: 1 / -1; }
  button.go {
    margin-top: 18px; width: 100%; padding: 13px; border: 0; border-radius: 12px; cursor: pointer;
    background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff;
    font-size: 1rem; font-weight: 600; transition: transform .1s, opacity .2s;
  }
  button.go:hover { opacity: .92; }
  button.go:active { transform: scale(.99); }
  button.go:disabled { opacity: .6; cursor: wait; }
  .placeholder { color: var(--muted); text-align: center; padding: 40px 10px; font-size: .92rem; }
  .price { font-size: 2.2rem; font-weight: 800; margin: 6px 0 2px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent; }
  .label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 16px 0; }
  .stat { background: var(--input); border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; }
  .stat b { display: block; font-size: .95rem; margin-top: 3px; }
  .explain { background: var(--ok-bg); border-radius: 12px; padding: 14px; line-height: 1.55; font-size: .93rem; }
  .note { color: var(--muted); font-size: .75rem; margin-top: 12px; line-height: 1.4; }
  .error { background: var(--err-bg); color: var(--err); border-radius: 12px; padding: 14px; font-size: .92rem; }
  footer { text-align: center; color: var(--muted); font-size: .78rem; margin-top: 28px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">🏠</div>
      <div>
        <h1>House Price Predictor</h1>
        <div class="sub">Instant estimate powered by a Linear Regression model</div>
      </div>
    </div>
    <button class="toggle" id="themeBtn" type="button" aria-label="Toggle theme">🌙 Dark</button>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Property details</h2>
      <p class="hint">Fill in the details below and get an estimated market price.</p>
      <div class="fields">
        <div class="full">
          <label for="Square_Footage">Square footage <small>(sq ft)</small></label>
          <input id="Square_Footage" type="number" min="100" step="1" placeholder="e.g. 2000">
        </div>
        <div>
          <label for="Num_Bedrooms">Bedrooms</label>
          <input id="Num_Bedrooms" type="number" min="0" step="1" placeholder="e.g. 3">
        </div>
        <div>
          <label for="Num_Bathrooms">Bathrooms</label>
          <input id="Num_Bathrooms" type="number" min="0" step="1" placeholder="e.g. 2">
        </div>
        <div>
          <label for="Year_Built">Year built</label>
          <input id="Year_Built" type="number" step="1" placeholder="e.g. 2005">
        </div>
        <div>
          <label for="Lot_Size">Lot size <small>(acres)</small></label>
          <input id="Lot_Size" type="number" min="0" step="0.01" placeholder="e.g. 1.5">
        </div>
        <div>
          <label for="Garage_Size">Garage size <small>(cars)</small></label>
          <input id="Garage_Size" type="number" min="0" step="1" placeholder="e.g. 2">
        </div>
        <div>
          <label for="Neighborhood_Quality">Neighbourhood quality <small>(1–10)</small></label>
          <input id="Neighborhood_Quality" type="number" min="1" max="10" step="1" placeholder="e.g. 7">
        </div>
      </div>
      <button class="go" id="goBtn" type="button">Predict price</button>
    </div>

    <div class="card">
      <h2>Estimated result</h2>
      <p class="hint">Your prediction will appear here.</p>
      <div id="result"><div class="placeholder">Enter the property details and press <b>Predict price</b>.</div></div>
    </div>
  </div>

  <footer>This is a statistical estimate, not a professional valuation.</footer>
</div>

<script>
  var FIELDS = ["Square_Footage","Num_Bedrooms","Num_Bathrooms","Year_Built","Lot_Size","Garage_Size","Neighborhood_Quality"];
  var root = document.documentElement;
  var themeBtn = document.getElementById("themeBtn");

  function syncBtn() {
    themeBtn.textContent = root.getAttribute("data-theme") === "dark" ? "☀️ Light" : "🌙 Dark";
  }
  syncBtn();
  themeBtn.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
    syncBtn();
  });

  var btn = document.getElementById("goBtn");
  var out = document.getElementById("result");

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  btn.addEventListener("click", async function () {
    var payload = {};
    FIELDS.forEach(function (f) { payload[f] = document.getElementById(f).value; });
    btn.disabled = true; btn.textContent = "Calculating…";
    try {
      var res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      var data = await res.json();
      if (!res.ok) {
        out.innerHTML = '<div class="error">⚠️ ' + esc(data.error || "Something went wrong.") + "</div>";
      } else {
        out.innerHTML =
          '<div class="label">Estimated price</div>' +
          '<div class="price">' + esc(data.price) + "</div>" +
          '<div class="stats">' +
            '<div class="stat"><span class="label">Likely range</span><b>' + esc(data.low) + " – " + esc(data.high) + "</b></div>" +
            '<div class="stat"><span class="label">Price / sq ft</span><b>' + esc(data.per_sqft) + "</b></div>" +
          "</div>" +
          '<div class="explain">' + esc(data.summary) + "</div>" +
          '<div class="note">The range is a rough ±10% guide around the estimate.</div>';
      }
    } catch (e) {
      out.innerHTML = '<div class="error">⚠️ Could not reach the server. Please try again.</div>';
    }
    btn.disabled = false; btn.textContent = "Predict price";
  });

  document.addEventListener("keydown", function (e) { if (e.key === "Enter") btn.click(); });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
