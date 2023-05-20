from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model2.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    x = int(request.form.get('x'))
    input_query = np.array([[x]])
    result=model.predict(input_query)[0]
    return (jsonify({'y':result}))

if __name__ == '__main__':
    app.run(debug=True)