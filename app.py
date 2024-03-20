from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the StandardScaler object
sc = pickle.load(open('sc.pkl', 'rb'))

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [float(x) for x in request.form.values()]
        if len(inputs) != expected_number_of_features:
            raise ValueError('Invalid number of input values')

        inputs = np.array([inputs])
        inputs = sc.transform(inputs)
        output = model.predict(inputs)

        # Convert output to 0 or 1 based on threshold
        prediction = int(output[0] >= 0.5)

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        error_message = f'Error: {str(e)}'
        return render_template('error.html', error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
