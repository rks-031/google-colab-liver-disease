from flask import Flask, request, url_for, render_template
import numpy as np
import pickle

app = Flask(__name__)

try:
    sc = pickle.load(open('sc.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading pickled files: {e}")


@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [int(x) for x in request.form.values()]
        inputs = np.array([inputs])
        inputs = sc.transform(inputs)
        output = model.predict(inputs)
        if output < 0.5:
            output = 0
        else:
            output = 1
        return render_template('result1.html', prediction=output)
    except Exception as e:
        print(f"Error processing prediction: {e}")
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)
