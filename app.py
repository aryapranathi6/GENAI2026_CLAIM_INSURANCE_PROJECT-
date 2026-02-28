from flask import Flask, render_template, request, redirect, session
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'tracers_secret'

# =========================
# CLIENT MODEL (DICT BUNDLE)
# =========================
client_bundle = pickle.load(open('models/client_fraud_model.pkl','rb'))

client_model = client_bundle["model"]
client_scaler = client_bundle["scaler"]
client_features = client_bundle["feature_columns"]
client_min = client_bundle["min_score"]
client_max = client_bundle["max_score"]

# =========================
# COMPANY MODEL (DIRECT MODEL)
# =========================
company_model = pickle.load(open('models/company_trust_model.pkl','rb'))

# XGBoost model expected features (detected from error)
COMPANY_FEATURE_COUNT = 54

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/client')
def client():
    return render_template('client.html')

@app.route('/company')
def company():
    return render_template('company.html')

# =========================
# LOGIN FLOW
# =========================
@app.route('/user_login', methods=['POST'])
def user_login():
    session['user_type'] = request.form['user_type']

    if session['user_type'] == 'client':
        return redirect('/client')
    else:
        return redirect('/company')

# =========================
# CLIENT PREDICTION
# =========================
@app.route('/predict_client', methods=['POST'])
def predict_client():

    data = [float(request.form[col]) for col in client_features]

    final = np.array([data])
    final = client_scaler.transform(final)

    prediction = client_model.predict(final)[0]

    score = (prediction - client_min) / (client_max - client_min) * 100

    return render_template('result.html', output=round(float(score),2))

# =========================
# COMPANY PREDICTION (SAFE FIX)
# =========================
@app.route('/predict_company', methods=['POST'])
def predict_company():

    data = [float(x) for x in request.form.values()]

    # Pad zeros to match expected 54 features
    while len(data) < COMPANY_FEATURE_COUNT:
        data.append(0)

    final = np.array([data])

    prediction = company_model.predict(final)[0]

    return render_template('result.html', output=round(float(prediction),2))

# =========================
# MAIN
# =========================
if __name__ == '__main__':
    app.run(debug=True)