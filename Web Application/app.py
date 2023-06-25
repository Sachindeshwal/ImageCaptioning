from flask import Flask, render_template, request
import numpy as np
import os
import pickle
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img,img_to_array

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'



image_model=InceptionV3(weights='imagenet')
image_model=Model(inputs=image_model.inputs,outputs=image_model.layers[-2].output)


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length=34



loaded_model = tf.keras.models.load_model('my_model.h5')

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequences
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=="POST":
        f=request.files['file']
        file_name=f.filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
        image = load_img(os.path.join(app.config['UPLOAD_FOLDER'], file_name),target_size=(299,299))
        image = img_to_array(image)
        image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        image = preprocess_input(image)
        feature = image_model.predict(image,verbose=0)
        print(type(feature))
        y_pred = predict_caption(loaded_model, feature, tokenizer, max_length)
        pred=" ".join(y_pred.split(" ")[1:][:-1])
        print(y_pred)
        return render_template('index.html',img='static/uploads/'+file_name, caption=pred)
    
    return render_template('index.html')

if __name__=='__main__':
    app.run()