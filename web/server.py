import os
from flask import Flask, render_template, request, send_from_directory
import src.predict_function_test as pf
import src.recommender_function as rf

app = Flask(__name__)

@app.route('/submit')
def index():
    return render_template('upload.html')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT,'data/prediction_images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        destination = '/'.join([target, filename])
        print(destination)
        file.save(destination)

    # return send_from_directory('images', filename, as_attachment=True)
    return render_template('complete.html', image_name = filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory('data/prediction_images',filename)

# @app.route('/test')
# def index2():
#     return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='8080',debug=True)
