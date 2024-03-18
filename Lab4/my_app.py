import pickle
import numpy as np
import os

from flask import Flask
from flask import request, render_template

reversed_label_mapping = {0: 'Bream',
 1: 'Parkki',
 2: 'Perch',
 3: 'Pike',
 4: 'Roach',
 5: 'Smelt',
 6: 'Whitefish'}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('./input.html')

@app.route("/predict", methods=["POST"])
def my_predict():
    weight = request.form['Weight']
    length1 = request.form['Length1']
    length2 = request.form['Length2']
    length3 = request.form['Length3']
    height = request.form['Height']
    width = request.form['Width']

    my_input = [weight, length1, length2, length3, height, width]
    clf3 = pickle.load(open("fish_predict_pipeline.pkl", "rb"))

    result = reversed_label_mapping[int(clf3.predict(np.array(my_input, dtype = float).reshape(1,6)))]
    print(result)
    return render_template('result.html', result=result)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)