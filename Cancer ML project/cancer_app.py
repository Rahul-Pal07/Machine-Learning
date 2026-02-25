from flask import Flask,render_template,request,redirect,url_for
import joblib
import numpy as np

model= joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    prediction= request.args.get("prediction")
    show_popup = True if prediction else False
    return render_template("cancerform.html",Prediction_text=prediction,show_popup=show_popup)

@app.route("/predict", methods=["POST"])
def predict():

    radius= float(request.form['radius'])
    texture= float(request.form["texture"])
    perimeter= float(request.form["perimeter"])
    area= float(request.form["area"])
    smoothness= float(request.form["smoothness"])

    features= np.array([[radius, texture, perimeter, area, smoothness]])

    # Scale features
    features_scaled = scaler.transform(features)


    prediction= model.predict(features_scaled)[0]

    result = "Malignant (Cancer)" if prediction == 1 else "Benign (No Cancer)"

    return redirect(url_for("home",prediction=result)+ "#popup")

if __name__ == "__main__":
    app.run(debug=True)