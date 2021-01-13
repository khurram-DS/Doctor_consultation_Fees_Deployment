
# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Gradient boosting regressor model
filename = 'model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Experience = int(request.form['Experience'])
        Rating = int(request.form['Rating'])
        Profile = int(request.form['Profile'])
        
        data = np.array([[Experience, Rating,Profile]])
        my_prediction = regressor.predict(data)
        output = round(my_prediction[0], 2)
        
        return render_template('index.html', prediction_text='Doctor Consulation fees is Rs. {}'.format(output))

if __name__ == '__main__':
	app.run(debug=True)