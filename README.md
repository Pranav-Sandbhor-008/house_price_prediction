# 🏡 House Price Prediction | Machine Learning & Flask

**Predict property prices with Machine Learning.** A modern, interactive web application built with Python, Flask, and Linear Regression that estimates property values from user-provided property details.

🚀 Deployment on Vercel

This project is deployed using Vercel, a cloud platform that makes deploying and hosting web applications simple and efficient.

🚀 **[Live Demo](https://house-price-prediction-two-topaz.vercel.app/)** · 💻 **[GitHub Repository](https://github.com/Pranav-Sandbhor-008/house_price_prediction)** · 👨‍💻 **[LinkedIn Profile](https://www.linkedin.com/in/pranavsandbhor/)**

---

## ✨ Project Overview

House Price Prediction is a machine learning web application that allows users to enter property characteristics and receive an estimated property value through a trained Linear Regression model.

The application combines a Python-based prediction backend with a modern, responsive interface designed to make machine learning predictions easy to understand and accessible across desktop and mobile devices.

## 🚀 Features

- 🧠 **Machine Learning Prediction:** Uses a trained Linear Regression model stored in `linear.pkl`.
- 🏠 **Property Analysis:** Accepts seven property-related input features.
- 🇮🇳 **Indian Currency Formatting:** Displays estimated prices in Indian Rupees (₹) with Indian number grouping.
- 📊 **Price Insights:** Calculates estimated price per square foot and property age.
- 🌗 **Light & Dark Themes:** Switch between themes for a comfortable user experience.
- 🎨 **Modern UI:** Gradient styling, responsive layouts, interactive controls, and polished result cards.
- ⚡ **Instant Predictions:** Generates an estimate when the user submits valid property details.
- 🧪 **Demo Inputs:** Pre-filled sample values help users explore the interface.
- 📥 **Downloadable Summary:** Download a text summary of the prediction and property information.
- ✅ **Input Validation:** Checks required values and permitted input ranges.
- ☁️ **Cloud Deployment:** Hosted on Vercel.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | Web application framework |
| Scikit-learn | Machine learning model |
| Linear Regression | Price prediction algorithm |
| NumPy | Numerical computations |
| HTML5 | Application structure |
| CSS3 | Responsive styling and themes |
| JavaScript | Theme switching and interactive features |
| Pickle | Serialization and loading of the trained model |
| Vercel | Cloud deployment |
| Git & GitHub | Version control and source code hosting |

## 🧠 Model Inputs

The model uses the following seven features:

| Feature | Description |
|---|---|
| `Square_Footage` | Total property area |
| `Num_Bedrooms` | Number of bedrooms |
| `Num_Bathrooms` | Number of bathrooms |
| `Year_Built` | Year of construction |
| `Lot_Size` | Size of the property lot |
| `Garage_Size` | Garage size |
| `Neighborhood_Quality` | Neighborhood quality score from 1 to 10 |

**Important:** Input units and feature order must match those used during model training for meaningful predictions.

## ⚙️ Getting Started

Follow these steps to run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/Pranav-Sandbhor-008/house_price_prediction.git
cd house_price_prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the model file

Make sure `linear.pkl` is present in the project root directory alongside `app.py`.

### 5. Run the application

```bash
python app.py
```

### 6. Open the application

Visit the following address in your browser:

`http://127.0.0.1:5000`

Enter the property details and click the prediction button to view the estimated value.

## 📁 Project Structure

```text
house_price_prediction/
│
├── app.py             # Flask application and UI
├── linear.pkl         # Trained Linear Regression model
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## 🔍 How It Works

1. **Enter Property Details:** The user provides the required property information.
2. **Validate Inputs:** The Flask application checks that the entered values are valid.
3. **Load the Model:** The trained model is loaded from `linear.pkl`.
4. **Generate Prediction:** The model processes the seven features and predicts a numerical value.
5. **Display Results:** The application displays the estimate and additional property insights.
6. **Download Summary:** The user can download a text report of the result.

## ☁️ Live Deployment

Try the application directly in your browser:

**[Launch House Price Prediction](https://house-price-prediction-two-topaz.vercel.app/)**

The application is deployed on Vercel, making it accessible online without requiring users to install Python or configure the project locally.

## ⚠️ Important Notes

- The prediction depends on the model's training data and performance.
- Currency formatting alone does not convert a USD-trained model into an INR-trained model. Confirm that the target values used during training are denominated in INR before interpreting predictions as Indian property prices.
- Estimated price per square foot is calculated from the predicted value and the entered area.
- Predictions are estimates, not guaranteed selling prices or professional property valuations.
- Only load trusted pickle files because deserializing untrusted pickle files can execute malicious code.
- Keep the model file and installed scikit-learn version compatible.

## 🔮 Future Enhancements

- 📍 Location-based property estimates
- 📈 Model performance and evaluation metrics
- 📊 Interactive price comparison charts
- 🗺️ Geographic property insights
- 🧠 Comparison of Linear Regression, Random Forest, and other algorithms
- 📄 PDF export for property reports
- 🏘️ Historical property-price analysis

## 👨‍💻 Author

**Pranav Sandbhor**

Computer Science Engineering | AI/ML | Data Science | Generative AI | Software Development

- 🔗 **LinkedIn:** [Pranav Sandbhor](https://www.linkedin.com/in/pranavsandbhor/)
- 💻 **GitHub:** [Pranav-Sandbhor-008](https://github.com/Pranav-Sandbhor-008)
- 🚀 **Live Project:** [House Price Prediction](https://house-price-prediction-two-topaz.vercel.app/)

## ⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub and sharing your feedback.

**Built with Python, Flask, and Machine Learning. Turning property data into useful insights!** 🏡
