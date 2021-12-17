# run by typing python3 main.py in a terminal 
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from ai import get_yolo_net, yolo_forward, yolo_save_img
from flask_utils import get_base_url, allowed_file, and_syntax

import torch

MODEL_PATH = 'best.pt'

# setup the webserver
'''
    coding center code
    port may need to be changed if there are multiple flask servers running on same server
    comment out below three lines of code when ready for production deployment
'''
port = 12396
# base_url = get_base_url(port)
# app = Flask(__name__, static_url_path=base_url+'static')

'''
    cv scaffold code
    uncomment below line when ready for production deployment
'''
app = Flask(__name__)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# set up yolo net for prediction
# you will need to change names, weights, and configuration files.
names_path = os.path.join('yolo', 'yolo.names')
LABELS = open(names_path).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# weights_path = os.path.join('yolo', 'yolo.weights')
# cfg_path = os.path.join('yolo', 'yolo.cfg')
# net = get_yolo_net(cfg_path, weights_path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)




@app.route('/')
# @app.route(base_url)
def home():
    return render_template('home.html')

@app.route('/about')
# @app.route(base_url + '/about')
def about():
    return render_template('about.html')

@app.route('/', methods=['POST'])
# @app.route(base_url, methods=['POST'])
def home_post():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('results', filename=filename))


@app.route('/uploads/<filename>')
# @app.route(base_url + '/uploads/<filename>')
def results(filename): 
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(image_path)
    results = model(image)
    print(dir(results))
    results_df = results.pandas().xywhn[0] #only one image
    print(results_df)
    
    # if we detected something
    if len(results_df.index) > 0:
        found = True
        new_filename = 'yolo_' + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        class_ids = list(results_df['class'])
        labels = list(results_df['name'])
        # boxes = results_df[[]]
        confidences = list(results_df['confidence'])
        
        # yolo_save_img(image, class_ids, boxes, labels, confidences, COLORS, file_path=file_path)
        results.save(file_path)
        

        # confidences: rounding and changing to percent, putting in function
        format_confidences = []
        for percent in confidences:
            format_confidences.append(str(round(percent*100)) + '%')
        format_confidences = and_syntax(format_confidences)
        # labels: sorting and capitalizing, putting into function
        labels = set(labels)
        labels = [plant.capitalize() for plant in labels]
        labels = and_syntax(labels)
        return render_template('results.html', confidences=format_confidences, labels=labels, 
            old_filename = filename, 
            filename=new_filename+'/image0.jpg') 
    else:
        found = False
        # replace 'Objects' with whatever you are trying to detect
        return render_template('results.html', labels='No Objects', old_filename=filename, filename=filename) 

@app.route('/files/<path:filename>')
# @app.route(base_url + '/files/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    # website_url = 'coding.ai-camp.org'
    # print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    # remove debug=True when deploying it
    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)

    '''
    cv scaffold code
    '''
    # Only for debugging while developing
#     app.run(port=port, debug=True)
