from flask import Flask,render_template,request,redirect,url_for
import joblib
import numpy as np

model= joblib.load("model.pkl")

app = Flask(__name__)


@app.route("/")
def hello_world():
    prediction = request.args.get("prediction")
    return render_template("myform.html", Prediction_text=prediction)


@app.route("/predict",methods=["POST"])
def predict():
    area = float(request.form["area"]) 
    bedroom = float(request.form["bedroom"])
    age = float(request.form["age"])

    feature = np.array([[area,bedroom,age]])
    prediction = model.predict(feature)[0]
    return redirect(url_for("hello_world", prediction=prediction) + "#popup")

if __name__=="__main__":
    app.run(debug=True)   