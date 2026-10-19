
import os
import pickle
from datetime import datetime

import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]

LABELS = {
    "Square_Footage": "Square footage",
    "Num_Bedrooms": "Bedrooms",
    "Num_Bathrooms": "Bathrooms",
    "Year_Built": "Year built",
    "Lot_Size": "Lot size",
    "Garage_Size": "Garage size",
    "Neighborhood_Quality": "Neighborhood quality",
}


def indian_number(value):
    """Format a number using Indian digit grouping."""
    value = f"{value:,.2f}"
    integer, decimal = value.split(".")
    if len(integer) > 3:
        last_three = integer[-3:]
        remaining = integer[:-3]
        groups = []
        while remaining:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        integer = ",".join(groups + [last_three])
    return f"{integer}.{decimal}"


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#635bff">
<title>PropValue India | Smart Property Estimator</title>
<style>
:root {
 --bg:#f5f7fc; --card:#fff; --text:#182338; --muted:#718096;
 --border:#e4e9f2; --input:#fff; --accent:#635bff;
 --soft:#efeeff; --shadow:0 16px 45px #1823380c;
}
body.dark {
 --bg:#0d1422; --card:#151f30; --text:#f0f4ff; --muted:#a5b1c5;
 --border:#2b374c; --input:#101827; --accent:#a29bff;
 --soft:#292744; --shadow:0 16px 45px #0003;
}
* {box-sizing:border-box}
body {
 margin:0;background:var(--bg);color:var(--text);
 font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;
 transition:background .2s,color .2s;
}
button,input {font:inherit}
button {cursor:pointer}
.nav {
 display:flex;align-items:center;justify-content:space-between;
 gap:12px;padding:16px max(5%,calc((100% - 1160px)/2));
 background:var(--card);border-bottom:1px solid var(--border);
}
.brand {display:flex;align-items:center;gap:11px;font-weight:850;font-size:19px}
.logo {
 width:42px;height:42px;border-radius:14px;display:grid;place-items:center;
 background:linear-gradient(135deg,#847bff,#5047df);color:white;font-size:24px;
}
.brand small {display:block;color:var(--muted);font-size:10px;letter-spacing:1px;margin-top:3px}
.nav-actions {display:flex;gap:8px}
.small-btn {
 border:1px solid var(--border);border-radius:11px;background:var(--card);
 color:var(--text);padding:10px 12px;font-size:12px;font-weight:750;
}
main {max-width:1160px;margin:auto;padding:38px 22px 50px}
.hero {display:flex;justify-content:space-between;align-items:center;gap:24px;margin-bottom:30px}
.eyebrow {
 display:inline-block;background:var(--soft);color:var(--accent);
 border-radius:30px;padding:8px 12px;font-size:11px;font-weight:850;letter-spacing:.6px
}
.hero h1 {font-size:clamp(30px,4vw,46px);letter-spacing:-1.6px;line-height:1.12;margin:15px 0 12px}
.gradient {background:linear-gradient(100deg,#847bff,#408edb);background-clip:text;-webkit-background-clip:text;color:transparent}
.hero p {color:var(--muted);line-height:1.8;font-size:14px;max-width:650px;margin:0}
.hero-chip {border:1px solid var(--border);background:var(--card);padding:15px 17px;border-radius:15px;min-width:150px}
.hero-chip b {display:block;font-size:14px;margin-bottom:5px}
.hero-chip span {color:var(--muted);font-size:11px}
.layout {display:grid;grid-template-columns:1.05fr .95fr;gap:22px;align-items:start}
.card {background:var(--card);border:1px solid var(--border);border-radius:21px;box-shadow:var(--shadow);overflow:hidden}
.card-head {padding:22px 24px;border-bottom:1px solid var(--border)}
.card-head h2 {font-size:18px;margin:0 0 7px}
.card-head p {font-size:12px;color:var(--muted);line-height:1.7;margin:0}
.form-body {padding:23px 24px 25px}
.section-title {font-size:12px;font-weight:850;margin:0 0 17px;display:flex;gap:9px;align-items:center}
.section-icon {width:30px;height:30px;display:grid;place-items:center;background:var(--soft);color:var(--accent);border-radius:9px}
.fields {display:grid;grid-template-columns:1fr 1fr;gap:17px 13px}
.field label {display:block;font-size:12px;font-weight:750;margin-bottom:8px}
.field input {width:100%;min-width:0;background:var(--input);border:1px solid var(--border);color:var(--text);border-radius:10px;padding:12px;outline:none}
.field input:focus {border-color:var(--accent);box-shadow:0 0 0 3px #635bff20}
.hint {display:block;color:var(--muted);font-size:10px;margin-top:6px;line-height:1.5}
.actions {display:grid;grid-template-columns:1fr auto;gap:9px;margin-top:23px}
.primary {
 border:0;border-radius:11px;padding:14px 12px;color:#fff;font-weight:850;font-size:13px;
 background:linear-gradient(110deg,#776dff,#5148df);box-shadow:0 8px 20px #635bff30;
}
.secondary {border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:11px;padding:12px;font-weight:750;font-size:12px}
.primary:hover,.secondary:hover,.small-btn:hover {filter:brightness(1.06);transform:translateY(-1px)}
.notice {text-align:center;color:var(--muted);font-size:10px;line-height:1.7;margin-top:13px}
.error {background:#fff0f0;color:#b42318;border:1px solid #ffd2d2;border-radius:11px;padding:13px;font-size:12px;margin-bottom:17px}
body.dark .error {background:#351d28;color:#ffb6bd;border-color:#59303b}
.result-top {padding:25px;color:white;background:radial-gradient(circle at 95% 5%,#aaa2ff77,transparent 34%),linear-gradient(135deg,#29265c,#4e49b8 65%,#7168ff)}
.result-tag {font-size:11px;color:#e1dfff;font-weight:850;letter-spacing:.7px}
.result-top h2 {font-size:21px;margin:13px 0 7px}
.result-top p {font-size:12px;color:#e2e2ff;line-height:1.7;margin:0}
.price-label {font-size:12px;color:#e2e2ff;margin-top:24px}
.price {font-size:clamp(27px,3.5vw,39px);font-weight:900;letter-spacing:-1px;overflow-wrap:anywhere;margin:5px 0}
.price-sub {font-size:11px;color:#e2e2ff}
.metrics {display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:20px 22px 0}
.metric {padding:15px;border:1px solid var(--border);border-radius:14px}
.metric span {display:block;font-size:10px;color:var(--muted);margin-bottom:9px}
.metric b {font-size:16px;overflow-wrap:anywhere}
.metric small {display:block;color:var(--muted);font-size:10px;margin-top:5px}
.result-body {padding:20px 22px 24px}
.result-body h3 {font-size:14px;margin:0 0 12px}
.explain {border:1px solid var(--border);background:var(--bg);border-radius:13px;padding:14px;color:var(--muted);font-size:12px;line-height:1.8}
.explain strong {color:var(--text)}
.pill {display:inline-block;background:var(--soft);color:var(--accent);padding:7px 10px;border-radius:25px;font-size:10px;font-weight:850;margin-top:13px}
.summary {margin-top:20px}
.summary-row {display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--border);padding:10px 0;font-size:11px}
.summary-row span {color:var(--muted)}
.summary-row b {text-align:right;overflow-wrap:anywhere}
.empty {padding:35px 25px;text-align:center}
.illustration {height:83px;width:83px;border-radius:24px;margin:0 auto 20px;display:grid;place-items:center;background:var(--soft);font-size:39px}
.empty h3 {margin:0 0 9px;font-size:18px}
.empty p {color:var(--muted);font-size:12px;line-height:1.8;max-width:300px;margin:0 auto}
.steps {display:grid;gap:12px;text-align:left;margin:25px auto 0;max-width:300px}
.step {display:flex;align-items:center;gap:11px;color:var(--muted);font-size:11px}
.step b {height:29px;width:29px;flex:0 0 29px;display:grid;place-items:center;background:var(--soft);color:var(--accent);border-radius:9px}
.tip {margin-top:22px;border:1px solid var(--border);border-radius:15px;padding:16px;display:flex;gap:12px;align-items:flex-start}
.tip-icon {font-size:22px}
.tip b {font-size:12px}
.tip p {font-size:11px;color:var(--muted);line-height:1.7;margin:5px 0 0}
.footer {text-align:center;color:var(--muted);font-size:10px;line-height:1.8;padding:27px 8px 0}
@media(max-width:820px) {
 .hero {align-items:flex-start;flex-direction:column}
 .hero-chip {display:none}
 .layout {grid-template-columns:1fr}
}
@media(max-width:460px) {
 main {padding:25px 13px 35px}
 .nav {padding:12px 13px}
 .brand {font-size:16px}
 .logo {height:37px;width:37px}
 .small-btn {padding:9px;font-size:11px}
 .card-head,.form-body {padding-left:17px;padding-right:17px}
 .fields {gap:15px 10px}
 .field input {padding:11px 9px;font-size:12px}
 .actions {grid-template-columns:1fr}
 .metrics {padding:15px 15px 0}
 .result-body {padding:18px 15px 20px}
 .result-top {padding:22px 18px}
}
</style>
</head>
<body>
<header class="nav">
 <div class="brand">
  <div class="logo">⌂</div>
  <div>PropValue India<small>SMART PROPERTY ESTIMATOR</small></div>
 </div>
 <div class="nav-actions">
  <button class="small-btn" type="button" onclick="fillSample()">✦ Try demo</button>
  <button class="small-btn" id="themeToggle" type="button">☾ Dark</button>
 </div>
</header>

<main>
 <section class="hero">
  <div>
   <div class="eyebrow">✦ MACHINE LEARNING · INDIA</div>
   <h1>Find your property's <span class="gradient">estimated value.</span></h1>
   <p>Turn property details into an easy-to-understand estimate. Enter your details, explore the insights, and get a result in seconds.</p>
  </div>
  <div class="hero-chip"><b>₹ INR estimates</b><span>Indian number formatting</span></div>
 </section>

 <div class="layout">
  <section class="card">
   <div class="card-head">
    <h2>🏡 Property details</h2>
    <p>Enter the details below. For best results, use the same units and scales as your model's training dataset.</p>
   </div>
   <div class="form-body">
    {% if error %}<div class="error" role="alert">⚠ {{ error }}</div>{% endif %}
    <form method="POST" id="propertyForm">
     <div class="section-title"><span class="section-icon">⌂</span> PROPERTY INFORMATION</div>
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
       <span class="hint">Number of bathrooms</span>
      </div>
      <div class="field">
       <label for="Year_Built">Year built</label>
       <input id="Year_Built" name="Year_Built" type="number" min="1800" max="{{ current_year }}" step="1" placeholder="e.g. 2015" value="{{ values.get('Year_Built','') }}" required>
       <span class="hint">Construction year</span>
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
     <div class="actions">
      <button class="primary" type="submit">✨ Calculate property estimate →</button>
      <button class="secondary" type="button" onclick="resetForm()">↺ Reset</button>
     </div>
     <div class="notice">Your property details are processed by this app to generate the estimate.</div>
    </form>
   </div>
  </section>

  <section class="card" aria-live="polite" id="resultPanel">
   {% if prediction is not none %}
    <div class="result-top">
     <div class="result-tag">✦ ESTIMATE GENERATED</div>
     <h2>Your estimated property value</h2>
     <p>Based on the information you entered and your trained ML model.</p>
     <div class="price-label">Predicted value (INR)</div>
     <div class="price">₹{{ prediction_display }}</div>
     <div class="price-sub">Indian Rupees · INR</div>
    </div>
    <div class="metrics">
     <div class="metric"><span>ESTIMATED PRICE / SQ FT</span><b>₹{{ per_sqft_display }}</b><small>Calculated from your area</small></div>
     <div class="metric"><span>PROPERTY AGE</span><b>{{ property_age }} years</b><small>Based on year built</small></div>
     <div class="metric"><span>BEDROOMS</span><b>{{ values.get('Num_Bedrooms','—') }}</b><small>Bedrooms entered</small></div>
     <div class="metric"><span>NEIGHBORHOOD SCORE</span><b>{{ values.get('Neighborhood_Quality','—') }}/10</b><small>Your supplied rating</small></div>
    </div>
    <div class="result-body">
     <h3>💡 Understand your result</h3>
     <div class="explain">
      Based on the details provided, the model estimates a property value of
      <strong>₹{{ prediction_display }}</strong>. The price per square foot is calculated by dividing this estimate by the entered square footage.
      This is a model-generated estimate, not a guaranteed selling price or formal valuation.
     </div>
     <div class="pill">✓ Prediction complete</div>
     <div class="summary">
      <h3 style="margin-top:20px">Your property snapshot</h3>
      {% for key in features %}
       <div class="summary-row"><span>{{ labels[key] }}</span><b>{{ values.get(key,'—') }}</b></div>
      {% endfor %}
     </div>
     <button class="primary" style="width:100%;margin-top:18px" type="button" onclick="downloadReport()">⬇ Download estimate summary</button>
    </div>
   {% else %}
    <div class="empty">
     <div class="illustration">🏠</div>
     <h3>Your estimate starts here</h3>
     <p>Enter property information or tap “Try demo” to load sample values. Your prediction and property insights will appear here.</p>
     <div class="steps">
      <div class="step"><b>01</b><span>Enter the property details</span></div>
      <div class="step"><b>02</b><span>Generate an ML-based estimate</span></div>
      <div class="step"><b>03</b><span>Explore your price insights</span></div>
     </div>
    </div>
    <div style="padding:0 22px 22px">
     <div class="tip"><div class="tip-icon">💡</div><div><b>Pro tip</b><p>Use realistic inputs and the same units as your training data. Sample values are only for testing the interface.</p></div></div>
    </div>
   {% endif %}
  </section>
 </div>
 <div class="footer">
  PropValue India · Built with Flask &amp; Linear Regression<br>
  Estimates are model-dependent and may differ from actual market prices. Not financial or real-estate advice.
 </div>
</main>

<script>
const toggle = document.getElementById("themeToggle");

function setTheme(theme) {
 document.body.classList.toggle("dark", theme === "dark");
 toggle.textContent = theme === "dark" ? "☀ Light" : "☾ Dark";
 try { localStorage.setItem("propvalue-theme", theme); } catch(e) {}
}
let initialTheme = "light";
try { initialTheme = localStorage.getItem("propvalue-theme") || "light"; } catch(e) {}
setTheme(initialTheme);

toggle.addEventListener("click", () => {
 setTheme(document.body.classList.contains("dark") ? "light" : "dark");
});

function fillSample() {
 const sample = {
  Square_Footage: 2000,
  Num_Bedrooms: 3,
  Num_Bathrooms: 2,
  Year_Built: 2015,
  Lot_Size: 0.25,
  Garage_Size: 2,
  Neighborhood_Quality: 8
 };
 for (const [key, value] of Object.entries(sample)) {
  document.getElementById(key).value = value;
 }
 document.getElementById("Square_Footage").focus();
}

function resetForm() {
 window.location.href = "/";
}

function downloadReport() {
 const lines = [
  "PropValue India - Property Estimate Summary",
  "===========================================",
  "Estimated value: ₹{{ prediction_display }}",
  "Estimated price per square foot: ₹{{ per_sqft_display }}",
  "Property age: {{ property_age }} years",
  "",
  "Property details:"
  {% for key in features %}
  , "{{ labels[key] }}: {{ values.get(key,'—') }}"
  {% endfor %}
  ,
  "",
  "Disclaimer: This is a model-generated estimate, not a formal valuation."
 ];
 const blob = new Blob([lines.join("\n")], {type:"text/plain;charset=utf-8"});
 const url = URL.createObjectURL(blob);
 const a = document.createElement("a");
 a.href = url;
 a.download = "property-estimate.txt";
 a.click();
 URL.revokeObjectURL(url);
}
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prediction_display = None
    per_sqft_display = None
    property_age = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()
                if not raw:
                    raise ValueError(
                        f"Please enter {LABELS[feature].lower()}."
                    )
                values[feature] = raw

            numbers = {key: float(values[key]) for key in FEATURES}

            if not all(np.isfinite(v) for v in numbers.values()):
                raise ValueError("Please enter valid numeric values.")

            current_year = datetime.now().year

            limits = {
                "Square_Footage": (1, 1_000_000),
                "Lot_Size": (0, 1_000_000),
                "Num_Bedrooms": (0, 100),
                "Num_Bathrooms": (0, 100),
                "Year_Built": (1800, current_year),
                "Garage_Size": (0, 100),
                "Neighborhood_Quality": (1, 10),
            }

            for key, (minimum, maximum) in limits.items():
                if not minimum <= numbers[key] <= maximum:
                    raise ValueError(
                        f"{LABELS[key]} must be between "
                        f"{minimum} and {maximum}."
                    )

            for key in ("Num_Bedrooms", "Year_Built"):
                if numbers[key] % 1 != 0:
                    raise ValueError(
                        f"{LABELS[key]} must be a whole number."
                    )

            # Feature order must match model training.
            row = np.array(
                [[numbers[key] for key in FEATURES]],
                dtype=float,
            )
            result = float(model.predict(row)[0])

            if not np.isfinite(result):
                raise ValueError("The model returned an invalid prediction.")

            # Display estimates as non-negative values.
            prediction = max(0.0, result)
            prediction_display = indian_number(prediction)
            per_sqft_display = indian_number(
                prediction / numbers["Square_Footage"]
            )
            property_age = current_year - int(numbers["Year_Built"])

        except (ValueError, TypeError, OverflowError) as exc:
            error = str(exc) or "Please check your inputs and try again."

    return render_template_string(
        HTML,
        prediction=prediction,
        prediction_display=prediction_display,
        per_sqft_display=per_sqft_display,
        property_age=property_age,
        error=error,
        values=values,
        features=FEATURES,
        labels=LABELS,
        current_year=datetime.now().year,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
