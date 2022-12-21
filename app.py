
from flask import Flask, request
import joblib
import numpy as np
import json

# URL https://vid-diabetes-prediction.herokuapp.com/

app = Flask(__name__)

@app.route("/")
def welcome():
    return "welcome Vidhi Patel"

@app.route("/diabetes_predict", methods=['POST'])
def diabetes_pred():
    model = joblib.load("./DIABETES_CLASSIFICATION_MODEL_98_ACC.joblib")
    
    pregnancies = request.form.get('pregnancies')
    glucose = request.form.get('glucose')
    bloodpressure = request.form.get('bloodpressure')
    skinthickness = request.form.get('skinthickness')
    insulin = request.form.get('insulin')
    bmi = request.form.get('bmi')
    dpf = request.form.get('dpf')
    age = request.form.get('age')
    
    input = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
    result = model.predict(input)[0]
    
    return json.dumps({'diabetes':str(result)})

if __name__ == "__main__":
    app.run(debug=True)