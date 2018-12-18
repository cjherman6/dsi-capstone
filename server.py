import os
from flask import Flask, render_template, request
#fastai dog breed predictor
import predict_function as pf
#scratch recommender function
import recommender_function as rf
import pickle

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, 'app_data/prediction_images')
filename = None

@app.route('/')
def index(): 
    return render_template('upload.html')

@app.route('/upload',methods = ['POST'])
def upload():

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        global filename
        filename = file.filename
        destination = '/'.join([target,filename])
        print(destination)
        file.save(destination)
    return render_template('complete.html')

@app.route('/prediction',methods = ['GET','POST'])
def prediction():
    
    destination = '/'.join([target,filename])
    return pf.pred_output(filename)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
