from flask import Flask, request, redirect, url_for
from flask import request
from werkzeug.utils import secure_filename
import requests
import json
import os
from datetime import datetime
import mysql.connector
from flask import send_from_directory
import glob
import tensorflow as tf, numpy as np
import cv2
import facenet

UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route("/face", methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
	sess = tf.Session()

	class Embedder:
	    IMAGE_SIZE = 160

	    def __init__(self, model='20180402-114759/20180402-114759.pb', *args, **kwargs):
	        facenet.load_model(model)

	        self.images_placeholder =      tf.get_default_graph().get_tensor_by_name("input:0")
	        self.embeddings =              tf.get_default_graph().get_tensor_by_name("embeddings:0")
	        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
	        self.embedding_size = self.embeddings.get_shape()[1]

	    def embed(self, img, do_prewhiten=True):
	        if type(img) == str:
	            images = facenet.load_image([img], False, False, IMAGE_SIZE)
	        elif type(img) == np.ndarray and img.ndim == 2: # if 1 channel image (grayscale)
	            w, h = img.shape
	            ret = np.empty((w, h, 3), dtype=np.uint8)
	            ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
	            img = ret
	        if do_prewhiten:
	            img = facenet.prewhiten(img)
	        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
	        images = img.reshape(1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
	        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
	        feature_vector = sess.run(self.embeddings, feed_dict=feed_dict)
	        return feature_vector
	e=Embedder()
	files = sorted(glob.glob('*.jpg'))
	b=np.zeros(shape=(len(files),512))
	for i in range(len(files)):
    	img=cv2.imread(files[i])
    	a=e.embed(img)
    	b[i]=a
	np.save('db.npy', b, allow_pickle=False)
	sess.close()
    app.run()
