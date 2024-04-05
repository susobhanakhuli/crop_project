import os
from flask import Flask,redirect,url_for,render_template,request
import pickle
import numpy as np

# to load pickle file
model=pickle.load(open('model.pkl','rb'))

# For image path
img = os.path.join('static', 'Images')

# WSGI Application
app=Flask(__name__)

# Decorator
@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/pred')
def predict():
    return render_template('pred.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


# Result HTML page
@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        int_features=[float(x) for x in request.form.values()]
        final=[np.array(int_features)]
        prediction = model.predict_proba(final)
        print(prediction)
        crop_dict = {
            20: 'rice',
            11: 'maize',
            3: 'chickpea',
            9: 'kidney beans',
            18: 'pigeon peas',
            13: 'moth beans',
            14: 'mung bean',
            2: 'black gram',
            10: 'lentil',
            19: 'pomegranate',
            1: 'banana',
            12: 'mango',
            7: 'grapes',
            21: 'watermelon',
            15: 'muskmelon',
            0: 'apple',
            16: 'orange',
            17: 'papaya',
            4: 'coconut',
            6: 'cotton',
            8: 'jute',
            5: 'coffee'
        }

        # print(crop_dict[prediction])
        count = 0
        mx_pred = 0
        mx_pred_count =0

        
        for i in prediction[0]:
            if i == 1:
                break
            else:
                if i>mx_pred:
                    mx_pred = i
                    mx_pred_count = count
            count += 1
        if count>21:
            count = mx_pred_count

        output = crop_dict[count]
        img_name = output+".png"
        file = os.path.join(img, img_name)
        return render_template('result.html', pred = [file, output])

# if __name__=='__main__':
#     app.run(debug=True)
    # app.run(debug=False, host='0.0.0.0')
