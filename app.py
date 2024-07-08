#flask,scikit-learn,pandas,pickle-mixin
import pandas as pd
from flask import Flask,request,jsonify,render_template,request
import pickle
import numpy as np
import csv

app = Flask(__name__)
# data = pd.read_csv('creditCardFraud_28011964_120214.csv')

with open('RandomForest.pkl', 'rb') as file:
    pipe = pickle.load(file)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
# def predict():
#     try:
#         # Get JSON data from request
#         data = request.get_json(force=True)

#         # Prepare the data for prediction (ensure order matches training data)
#         feature_values = [float(data[field]) for field in data]

#         # Predict using the loaded model
#         prediction = pipe.predict([feature_values])

#         # Return the result
#         result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
#         return jsonify(result=result)
#     except Exception as e:
#         return jsonify(result=str(e))

def predict():
    try:
        data = request.get_json(force=True)

        features = [
            data.get('sex'),
            data.get('education'),
            data.get('marriage'),
            data.get('age'),
            data.get('limit_bal'),
            data.get('pay_1'),
            data.get('pay_2'),
            data.get('pay_3'),
            data.get('pay_4'),
            data.get('pay_5'),
            data.get('pay_6'),
            data.get('bill_amt1'),
            data.get('bill_amt2'),
            data.get('bill_amt3'),
            data.get('bill_amt4'),
            data.get('bill_amt5'),
            data.get('bill_amt6'),
            data.get('pay_amt1'),
            data.get('pay_amt2'),
            data.get('pay_amt3'),
            data.get('pay_amt4'),
            data.get('pay_amt5'),
            data.get('pay_amt6'),
            


        # sex == int(request.form.get('sex')),
        # education == int(request.form.get('education')),
        # marriage == int(request.form.get('marriage')),
        # age == int(request.form.get('age')),
        # limit_bal == float(request.form.get('limit_bal')),
        # pay_1 == int(request.form.get('pay_1')),
        # pay_2 == int(request.form.get('pay_2')),
        # pay_3 == int(request.form.get('pay_3')),
        # pay_4 == int(request.form.get('pay_4')),
        # pay_5 == int(request.form.get('pay_5')),
        # pay_6 == int(request.form.get('pay_6')),
        # bill_amt1 == float(request.form.get('bill_amt1')),
        # bill_amt2 == float(request.form.get('bill_amt2')),
        # bill_amt3 == float(request.form.get('bill_amt3')),
        # bill_amt4 == float(request.form.get('bill_amt4')),
        # bill_amt5 == float(request.form.get('bill_amt5')),
        # bill_amt6 == float(request.form.get('bill_amt6')),
        # pay_amt1 == float(request.form.get('pay_amt1')),
        # pay_amt2 == float(request.form.get('pay_amt2')),
        # pay_amt3 == float(request.form.get('pay_amt3')),
        # pay_amt4 == float(request.form.get('pay_amt4')),
        # pay_amt5 == float(request.form.get('pay_amt5')),
        # pay_amt6 == float(request.form.get('pay_amt6')),
        # print(sex,education,marriage,age,limit_bal,pay_1,pay_2,pay_3,pay_4,pay_5,pay_6,bill_amt1,bill_amt2,bill_amt3,bill_amt4,bill_amt5,bill_amt6,
        # pay_amt1,pay_amt2,pay_amt3,pay_amt4,pay_amt5,pay_amt6)
        ]
        # Ensure all values are converted to floats
        features = [float(value) for value in features]

        # Convert the features into a numpy array and reshape for prediction
        final_features = np.array(features).reshape(1, -1)

        # Make the prediction
        prediction = pipe.predict(final_features)

        # Return the prediction as a JSON response
        #Return the result
        # result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
        
        results = ""
        if prediction == 1:
            results = "The credit card holder will be Defaulter in the next month"
        else:
            results = "The Credit card holder will not be Defaulter in the next month"
        return jsonify(result=results)

        # print(sex,education,marriage,age,limit_bal,pay_1,pay_2,pay_3,pay_4,pay_5,pay_6,bill_amt1,bill_amt2,bill_amt3,bill_amt4,bill_amt5,bill_amt6,
        # pay_amt1,pay_amt2,pay_amt3,pay_amt4,pay_amt5,pay_amt6)
        # input = pd.DataFrame([[sex,education,marriage,age,limit_bal,pay_1,pay_2,pay_3,pay_4,pay_5,pay_6,bill_amt1,bill_amt2,bill_amt3,bill_amt4,bill_amt5,bill_amt6,
        # pay_amt1,pay_amt2,pay_amt3,pay_amt4,pay_amt5,pay_amt6]],columns = ['sex','education','marriage','age','limit_bal','pay_1','pay_2','pay_3','pay_4','pay_5','pay_6','bill_amt1','bill_amt2','bill_amt3','bill_amt4','bill_amt5','bill_amt6',
        # 'pay_amt1','pay_amt2','pay_amt3','pay_amt4','pay_amt5','pay_amt6'])
        # prediction = pipe.predict(input)
        # print("Prediction Value:",prediction)
        # results = ""
        # if prediction == 1:
        #    results = "The credit card holder will be Defaulter in the next month"
        # else:
        #     results = "The Credit card holder will not be Defaulter in the next month"

        # return jsonify(result = results)

    except Exception as e:
        return jsonify({'error': str(e)})


    

if __name__=="__main__":
    app.run(debug=True,port=5006)