from flask import Flask,render_template,request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('joblib_model')
transformer = joblib.load('joblib_tranformer')


@app.route('/')

def home():
     return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
     try:
          
        year = request.form.get('year')
        make = request.form.get('make')
        model_ = request.form.get('model')
        trim = request.form.get('trim')
        body = request.form.get('body')
        transmission = request.form.get('transmission')
        odometer = request.form.get('odometer')
        color = request.form.get('color')
        interior = request.form.get('interior')
        condition = request.form.get('condition')
        mmr = request.form.get('mmr')
        
          
        input_dict = {
          'year': [year],
          'make': [make],
          'model': [model_],
          'trim': [trim],
          'body': [body],
          'transmission': [transmission],
          'odometer': [odometer],
          'color': [color],
          'interior': [interior],
          'condition': [condition],
          'mmr':[mmr] 
        }
        input_df = pd.DataFrame(input_dict)
        tranformer_data = transformer.transform(input_df)
        prediction = model.predict(tranformer_data)
        return render_template('index.html',prediction = prediction)
   
     except Exception as e:
        return render_template('index.html', error=str(e))
   
   
if __name__ == '__main__':
    app.run(debug=True)        


