from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

bestmodel = pickle.load(open('heart-risk-best.pkl','rb'))

heart_risk_scaler = pickle.load(open('heart-risk-scaler.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return "Hello World"
@app.route('/predict' , methods=['POST'])
def predict():
    Age=request.form.get('Age')
    Sex=request.form.get('Sex')
    ChestPainType=request.form.get('ChestPainType')
    RestingBP=request.form.get('RestingBP')
    Cholesterol=request.form.get('Cholesterol')
    FastingBS=request.form.get('FastingBS')
    RestingECG=request.form.get('RestingECG')
    MaxHR=request.form.get('MaxHR')
    ExerciseAngina=request.form.get('ExerciseAngina')
    Oldpeak=request.form.get('Oldpeak')
    ST_Slope=request.form.get('ST_Slope')

    # results={'Age':Age,'Sex':Sex,'ChestPainType':ChestPainType,'RestingBP':RestingBP,
    #          'Cholesterol':Cholesterol,'FastingBS':FastingBS,'RestingECG':RestingECG,
    #          'MaxHR':MaxHR,'ExerciseAngina':ExerciseAngina,'Oldpeak':Oldpeak,'ST_Slope':ST_Slope}


    input_query = np.array([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,
                             RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])

    # testingdata = pd.DataFrame([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
    #                              RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
    inumpyarray = np.asarray(input_query)
    testingdata = inumpyarray.reshape(1, -1)
    scaled_features = heart_risk_scaler.transform(testingdata)

    results=bestmodel.predict(scaled_features)[0]


    return jsonify({'results': str(results)})


if __name__== '__main__':
    app.run(debug=True)