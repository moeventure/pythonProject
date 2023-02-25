from flask import Flask, request, jsonify
import pickle
import numpy as np

model1 = pickle.load(open('modelrfrExpenses.pkl', 'rb'))
model2 = pickle.load(open('modelrfrFinal.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/forecast', methods=['POST'])
def predict():
    income = request.form.get("income")
    totalIncome = request.form.get("totalincome")
    totalExpenses = request.form.get("totalexpenses")

    totalSavings = (float(totalIncome) - float(totalExpenses)) / float(totalIncome) * 100
    totalSavings = "%.2f" % totalSavings

    input_query1 = np.array([[income, totalSavings]], dtype=float)
    input_query2 = np.array([[totalExpenses, totalIncome, totalSavings]], dtype=float)

    result1 = model1.predict(input_query1)[0]
    result1 = "{:.2f}".format(float(result1))

    result2 = model2.predict(input_query2)[0]
    result2 = "{:.2f}".format(float(result2))

    return jsonify({'placement': str(result1),
                    'placement2': str(result2)})


if __name__ == '__main__':
    app.run(debug=True)
