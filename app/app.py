from flask import Flask, render_template
app = Flask(__name__)
import os
import tensorflow as tf
from tensorflow import keras
# mode = 'C:\\Projects\\Bhadke Image Classification CNN\\computer_vision\\basedata\\model_v1'
mode_1 = 'C:\\Projects\\Bhadke Image Classification CNN\\phase 2\\models\\model_v3'
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model(mode_1)
# Show the model architecture
new_model.summary()
import PIL
import numpy as np
img_height=600
img_width = 400
class_names=['paddy', 'weed']
path = 'C:\\Projects\\Bhadke Image Classification CNN\\computer_vision\\basedata\\test\\'


def predict(path_file):
    # file = 'cuperus irri.jpg'
    # print(file)
    # path_file = path + file
    img = keras.preprocessing.image.load_img(
        path_file, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
#     display(PIL.Image.open(str(path_file)))
    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    template_data = {'prediction':class_names[np.argmax(score)], 'score':100 * np.max(score)}
    return template_data


# @app.route('/')
# def base():
# 	return render_template('file.html')
    # return('''<html>
    # <h1> CROP CLASSIFIER</h1><br><br><br>
    # <h3>Upload a crop image</h3>
    # <br><br><br>
    # <h3>Prediction</h3>
    # <h6></h6>
    # </html>
    # ''')


import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	prediction = {'prediction':'', 'score':''}
	return render_template('upload.html', predictions=prediction)

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed')
		path_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		prediction = predict(path_file)
		return render_template('upload.html', filename=filename, predictions=prediction)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
# In[9]:


app.run(debug=True)


# In[ ]:




