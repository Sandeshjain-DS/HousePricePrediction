#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#The next part was to make an API which receives sales details through GUI and computes the predicted sales value based on our mode

#import required library
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle #(pickling is a way to convert a python objects to character stream)

#we are instantiating a Flask object by passing __name__ argument to the Flask constructor.
#The Flask constructor has one required argument which is the name of the application package
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#A Route is an act of binding a URL to a view function.
@app.route('/')
def home():
    return render_template('index.html')# I set the main page using index.html

# On submitting the form values using POST request to /predict, we get the predicted sales value
@app.route('/predict',methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted cost is $ {}'.format(output))

#the results can be shown by making another POST request
#to request a response from the server, there are mainly two methods:
#GET : to request data from the server.
#POST : to submit data to be processed to the server.

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    print(data)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run()


# In[ ]:





# In[ ]:




