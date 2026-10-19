
import os
import pickle
from datetime import datetime

import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Linear Regression model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PropValue | Property Price Predictor</title>
<meta name="theme-color" content="#101827">

<style>
:root {
  --bg: #f4f7fc;
  --card: #ffffff;
  --text: #17243a;
  --muted: #718096;
  --border: #e3e9f2;
  --input: #ffffff;
  --accent: #625cf5;
  --accent2: #4b45d9;
  --soft: #eeedff;
  --shadow: 0 18px 55px rgba(31, 45, 80, .09);
}
body.dark {
  --bg: #0c1220;
  --card: #151e2e;
  --text: #edf2ff;
  --muted: #a1aec4;
  --border: #29354a;
  --input: #101827;
  --accent: #948dff;
  --accent2: #aaa5ff;
  --soft: #282744;
  --shadow: 0 18px 55px rgba(0, 0, 0, .24);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system,
               BlinkMacSystemFont, "Segoe UI", sans-serif;
  transition: background .25s, color .25s;
}
button, input { font: inherit; }
button { cursor: pointer; }
.nav {
  display: flex; justify-content: space-between; align-items: center;
  padding: 20px max(5%, calc((100% - 1120px) / 2));
  border-bottom: 1px solid var(--border);
  background: var(--card);
}
.brand { display:flex; align-items:center; gap:11px; font-weight:800; font-size:20px; }
.logo {
  width:42px; height:42px; border-radius:13px; display:grid;
  place-items:center; color:white; font-size:22px;
  background:linear-gradient(135deg,#756cff,#4740d8);
  box-shadow:0 6px 16px #625cf540;
}
.brand small { display:block; font-size:11px; font-weight:500; color:var(--muted); margin-top:2px; }
.theme-btn {
  border:1px solid var(--border); background:var(--card); color:var(--text);
  border-radius:12px; padding:10px 14px; font-size:13px; font-weight:650;
}
main { max-width:1120px; margin:auto; padding:46px 22px 60px; }
.hero { text-align:center; max-width:720px; margin:0 auto 32px; }
.eyebrow {
  display:inline-flex; align-items:center; gap:8px;
  color:var(--accent); background:var(--soft); border-radius:30px;
  padding:8px 13px; font-size:12px; font-weight:800; letter-spacing:.5px;
}
.hero h1 { font-size:clamp(32px,5vw,49px); letter-spacing:-1.8px; line-height:1.1; margin:20px 0 14px; }
.hero h1 span {
  background:linear-gradient(100deg,#7b73ff,#4c8fe8);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero p { color:var(--muted); line-height:1.8; font-size:15px; margin:0; }
.layout { display:grid; grid-template-columns:1.08fr .92fr; gap:24px; align-items:start; }
.card {
  background:var(--card); border:1px solid var(--border);
  border-radius:22px; box-shadow:var(--shadow); overflow:hidden;
}
.card-head { padding:24px 26px 18px; border-bottom:1px solid var(--border); }
.card-head h2 { margin:0 0 7px; font-size:19px; }
.card-head p { margin:0; font-size:13px; color:var(--muted); line-height:1.6; }
.form-body { padding:24px 26px 26px; }
.section-title {
  display:flex; align-items:center; gap:9px; font-size:13px;
  font-weight:800; margin:0 0 18px;
}
.section-icon {
  width:30px; height:30px; display:grid; place-items:center;
  background:var(--soft); color:var(--accent); border-radius:9px;
}
.fields { display:grid; grid-template-columns:1fr 1fr; gap:18px 14px; }
.field label { display:block; font-size:12px; font-weight:700; margin-bottom:8px; }
.field input {
  width:100%; min-width:0; padding:13px 12px;
  background:var(--input); border:1px solid var(--border);
  color:var(--text); border-radius:11px; outline:none;
  transition:border .2s, box-shadow .2s;
}
.field input:focus { border-color:var(--accent); box-shadow:0 0 0 3px #625cf51a; }
.hint { display:block; color:var(--muted); font-size:11px; margin-top:6px; }
.submit {
  width:100%; border:0; border-radius:12px; padding:15px;
  margin-top:25px; color:white; font-weight:800; font-size:14px;
  background:linear-gradient(110deg,#7168ff,#5048df);
  box-shadow:0 9px 22px #625cf52c; transition:transform .2s, filter .2s;
}
.submit:hover { transform:translateY(-2px); filter:brightness(1.06); }
.notice { margin-top:15px; color:var(--muted); font-size:11px; line-height:1.6; text-align:center; }
.error {
  background:#fff0f0; color:#b42318; border:1px solid #ffd2d2;
  border-radius:11px; padding:13px; font-size:13px; margin-bottom:18px;
}
body.dark .error { background:#351d28; color:#ffb6bd; border-color:#59303b; }
.result-card { min-height:100%; }
.result-top {
  padding:27px; color:white;
  background:radial-gradient(circle at 90% 5%,#a59eff88,transparent 34%),
             linear-gradient(135deg,#28275c,#4e49b8 65%,#6962ef);
}
.result-top .mini { font-size:12px; color:#e0deff; font-weight:700; }
.result-top h2 { font-size:22px; margin:13px 0 8px; }
.result-top p { color:#e1e3ff; line-height:1.7; font-size:13px; margin:0; }
.price-label { font-size:12px; color:#e0deff; margin-top:26px; }
.price { font-size:clamp(29px,4vw,39px); letter-spacing:-1px; font-weight:850; margin:7px 0 0; overflow-wrap:anywhere; }
.result-body { padding:25px 27px 28px; }
.result-body h3 { font-size:15px; margin:0 0 16px; }
.explain {
  border:1px solid var(--border); border-radius:14px; padding:16px;
  color:var(--muted); font-size:13px; line-height:1.8;
}
.explain strong { color:var(--text); }
.pill {
  display:inline-flex; margin-top:17px; padding:7px 10px;
  border-radius:20px; background:var(--soft); color:var(--accent);
  font-size:11px; font-weight:800;
}
.empty {
  padding:34px 27px 35px; text-align:center;
}
.illustration {
  width:90px; height:90px; margin:4px auto 22px; border-radius:25px;
  display:grid; place-items:center; font-size:40px;
  background:linear-gradient(135deg,var(--soft),var(--bg));
  border:1px solid var(--border);
}
.empty h3 { font-size:18px; margin:0 0 10px; }
.empty p { font-size:13px; color:var(--muted); line-height:1.8; margin:0 auto; max-width:290px; }
.steps { display:grid; gap:13px; text-align:left; margin-top:27px; }
.step { display:flex; gap:12px; align-items:center; font-size:12px; color:var(--muted); }
.step b {
  flex:0 0 30px; height:30px; display:grid; place-items:center;
  border-radius:10px; background:var(--soft); color:var(--accent);
}
.footer { text-align:center; padding:28px 12px 0; color:var(--muted); font-size:11px; line-height:1.8; }
@media(max-width:800px) {
  main { padding:32px 16px 40px; }
  .layout { grid-template-columns:1fr; }
  .hero { margin-bottom:25px; }
}
@media(max-width:430px) {
  .nav { padding:14px 16px; }
  .brand { font-size:17px; }
  .logo { width:37px; height:37px; }
  .theme-btn { padding:9px 10px; }
  .card-head,.form-body { padding-left:19px; padding-right:19px; }
  .fields { gap:16px 10px; }
  .field input { padding:12px 9px; font-size:13px; }
  .result-top,.result-body { padding:22px 20px; }
}
</style>
</head>
<body>
<header class="nav">
  <div class="brand">
    <div class="logo">⌂</div>
    <div>PropValue<small>SMART PROPERTY ESTIMATES</small></div>
  </div>
  <button class="theme-btn" id="themeToggle" type="button" aria-label="Toggle color theme">
    ☾ Dark mode
  </button>
</header>

<main>
  <section class="hero">
    <div class="eyebrow">✦ AI-POWERED ESTIMATION</div>
    <h1>Discover what your property <span>could be worth.</span></h1>
    <p>Enter your property's details and let our machine learning model estimate its value in seconds. Simple inputs, useful insights.</p>
  </section>

  <div class="layout">
    <section class="card">
      <div class="card-head">
        <h2>Property details</h2>
        <p>Fill in the information below to generate your personalized estimate.</p>
      </div>
      <div class="form-body">
        {% if error %}
          <div class="error" role="alert">⚠ {{ error }}</div>
        {% endif %}

        <form method="POST" action="/">
          <div class="section-title"><span class="section-icon">⌂</span> Property information</div>
          <div class="fields">
            <div class="field">
              <label for="Square_Footage">Square footage</label>
              <input id="Square_Footage" name="Square_Footage" type="number" min="1" max="1000000" step="any" placeholder="e.g. 2000" value="{{ values.get('Square_Footage','') }}" required>
              <span class="hint">Total interior area</span>
            </div>
            <div class="field">
              <label for="Lot_Size">Lot size</label>
              <input id="Lot_Size" name="Lot_Size" type="number" min="0" max="1000000" step="any" placeholder="e.g. 0.25" value="{{ values.get('Lot_Size','') }}" required>
              <span class="hint">Use your dataset's unit</span>
            </div>
            <div class="field">
              <label for="Num_Bedrooms">Bedrooms</label>
              <input id="Num_Bedrooms" name="Num_Bedrooms" type="number" min="0" max="100" step="1" placeholder="e.g. 3" value="{{ values.get('Num_Bedrooms','') }}" required>
              <span class="hint">Number of bedrooms</span>
            </div>
            <div class="field">
              <label for="Num_Bathrooms">Bathrooms</label>
              <input id="Num_Bathrooms" name="Num_Bathrooms" type="number" min="0" max="100" step="any" placeholder="e.g. 2" value="{{ values.get('Num_Bathrooms','') }}" required>
              <span class="hint">Full or partial, as applicable</span>
            </div>
            <div class="field">
              <label for="Year_Built">Year built</label>
              <input id="Year_Built" name="Year_Built" type="number" min="1800" max="{{ current_year }}" step="1" placeholder="e.g. 2015" value="{{ values.get('Year_Built','') }}" required>
              <span class="hint">Original construction year</span>
            </div>
            <div class="field">
              <label for="Garage_Size">Garage size</label>
              <input id="Garage_Size" name="Garage_Size" type="number" min="0" max="100" step="any" placeholder="e.g. 2" value="{{ values.get('Garage_Size','') }}" required>
              <span class="hint">Use the dataset's definition</span>
            </div>
            <div class="field" style="grid-column:1/-1">
              <label for="Neighborhood_Quality">Neighborhood quality (1–10)</label>
              <input id="Neighborhood_Quality" name="Neighborhood_Quality" type="number" min="1" max="10" step="any" placeholder="e.g. 8" value="{{ values.get('Neighborhood_Quality','') }}" required>
              <span class="hint">1 = lower quality · 10 = higher quality</span>
            </div>
          </div>
          <button class="submit" type="submit">✦ Estimate my property value →</button>
          <div class="notice">Your estimate is generated by a trained Linear Regression model.</div>
        </form>
      </div>
    </section>

    <section class="card result-card" aria-live="polite">
      {% if prediction is not none %}
        <div class="result-top">
          <div class="mini">✦ YOUR ESTIMATE IS READY</div>
          <h2>Here is your predicted property value</h2>
          <p>Based on the property details you provided.</p>
          <div class="price-label">Estimated value</div>
          <div class="price">{{ currency }}{{ "{:,.2f}".format(prediction) }}</div>
        </div>
        <div class="result-body">
          <h3>What this means</h3>
          <div class="explain">
            Based on the information entered, the model estimates a property value of
            <strong>{{ currency }}{{ "{:,.2f}".format(prediction) }}</strong>.
            This is a mathematical estimate, not a guaranteed selling price or a formal appraisal.
          </div>
          <div class="pill">✓ Prediction generated successfully</div>
        </div>
      {% else %}
        <div class="empty">
          <div class="illustration">⌂</div>
          <h3>Your estimate will appear here</h3>
          <p>Complete the property form and select the estimate button. Your predicted value and a plain-language explanation will show up in this panel.</p>
          <div class="steps">
            <div class="step"><b>01</b><span>Enter your property information</span></div>
            <div class="step"><b>02</b><span>Let the ML model analyze your inputs</span></div>
            <div class="step"><b>03</b><span>Review your estimated property value</span></div>
          </div>
        </div>
      {% endif %}
    </section>
  </div>

  <div class="footer">
    PropValue · Built with Flask &amp; Machine Learning<br>
    Estimates depend on the model and training data. Actual market values may differ.
  </div>
</main>

<script>
const body = document.body;
const toggle = document.getElementById("themeToggle");

function setTheme(theme) {
  body.classList.toggle("dark", theme === "dark");
  toggle.textContent = theme === "dark" ? "☀ Light mode" : "☾ Dark mode";
  try { localStorage.setItem("propvalue-theme", theme); } catch (e) {}
}
let savedTheme = "light";
try { savedTheme = localStorage.getItem("propvalue-theme") || "light"; } catch (e) {}
setTheme(savedTheme);
toggle.addEventListener("click", () => {
  setTheme(body.classList.contains("dark") ? "light" : "dark");
});
</script>
</body>
</html>
"""


FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            # Read and validate all required features
            for feature in FEATURES:
                raw_value = request.form.get(feature, "").strip()
                if not raw_value:
                    raise ValueError(f"Please enter a value for {feature.replace('_', ' ')}.")
                values[feature] = raw_value

            numbers = {key: float(values[key]) for key in FEATURES}

            if not all(np.isfinite(value) for value in numbers.values()):
                raise ValueError("Please enter valid, finite numbers.")

            current_year = datetime.now().year

            if not 1 <= numbers["Square_Footage"] <= 1_000_000:
                raise ValueError("Square footage must be between 1 and 1,000,000.")
            if not 0 <= numbers["Lot_Size"] <= 1_000_000:
                raise ValueError("Lot size must be between 0 and 1,000,000.")
            if not 0 <= numbers["Num_Bedrooms"] <= 100:
                raise ValueError("Bedrooms must be between 0 and 100.")
            if numbers["Num_Bedrooms"] % 1 != 0:
                raise ValueError("Bedrooms must be a whole number.")
            if not 0 <= numbers["Num_Bathrooms"] <= 100:
                raise ValueError("Bathrooms must be between 0 and 100.")
            if not 1800 <= numbers["Year_Built"] <= current_year:
                raise ValueError(f"Year built must be between 1800 and {current_year}.")
            if numbers["Year_Built"] % 1 != 0:
                raise ValueError("Year built must be a whole number.")
            if not 0 <= numbers["Garage_Size"] <= 100:
                raise ValueError("Garage size must be between 0 and 100.")
            if not 1 <= numbers["Neighborhood_Quality"] <= 10:
                raise ValueError("Neighborhood quality must be between 1 and 10.")

            # Keep the same feature order used during model training
            input_data = np.array(
                [[numbers[feature] for feature in FEATURES]],
                dtype=float,
            )

            result = float(model.predict(input_data)[0])

            if not np.isfinite(result):
                raise ValueError("The model returned an invalid result.")

            # A negative property price is not a useful estimate
            prediction = max(0.0, result)

        except (ValueError, TypeError, OverflowError) as exc:
            error = str(exc) or "Please check your entries and try again."

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error,
        values=values,
        current_year=datetime.now().year,
        currency="$",
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
