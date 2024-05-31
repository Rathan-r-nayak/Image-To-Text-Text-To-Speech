from keras.preprocessing.sequence import pad_sequences
import os
import fnmatch
import cv2
import numpy as np
import string
import time
import sys
from keras.layers import Dense, LSTM, Reshape,Lambda, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional,Dropout
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional,Dropout
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras
import pytesseract


char_list = string.ascii_letters+string.digits

train_label_length = np.load('./static/wgt/train_label_length.npy')
train_input_length = np.load('./static/wgt/train_input_length.npy')
training_img = np.load('./static/wgt/training_img.npy')
train_padded_txt = np.load('./static/wgt/train_padded_txt.npy')
max_label_len = np.load('./static/wgt/max_label_len.npy')
inputs = Input(shape=(32,128,1))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# convolution layer with kernel size (3,3)
conv_1 = Conv2D(16, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(s)
conv_1 = Dropout(0.25)(conv_1)
conv_1 = Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(conv_1)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(32, (3,3), activation = 'relu',kernel_initializer='he_normal' , padding='same')(pool_1)
conv_2= BatchNormalization(axis=-1)(conv_2)
conv_2 = Dropout(0.25)(conv_2)
conv_2 = Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(conv_2)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(pool_2)
conv_3= BatchNormalization(axis=-1)(conv_3)
conv_3 = Dropout(0.25)(conv_3)
conv_3 = Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(conv_3)
conv_4 = Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal' ,padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
    
conv_5 = Conv2D(256, (3,3), activation = 'relu',kernel_initializer='he_normal' , padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
    
conv_6 = Conv2D(256, (3,3), activation = 'relu',kernel_initializer='he_normal' , padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
    
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    
# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
    
outputs = Dense(62+1, activation = 'softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)





def displayResults(x):
    li = []
    # load the saved best model weights
    gray_image = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        
        # Resize the grayscale image
    x = cv2.resize(gray_image, (128, 32))
    act_model.load_weights('./static/models/best_model.hdf5')
    prediction = act_model.predict(x.reshape(1,32,128,1))

    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                            greedy=True)[0][0])
    x = x.reshape(32,128)
    #   plt.title('Input Image')
    #   plt.imshow(x)
    #   plt.axis('off')
    #   plt.show()

    # see the results
    for x in out:
        # print("predicted text = ", end = '')
        for p in x:  
            if int(p) != -1:
                # print(char_list[int(p)], end = '')
                li.append(char_list[int(p)])
    return li


def detect_and_crop_text(image_path):
    # Load the image
    # image = cv2.imread(image_path)
    image = np.asarray(image_path)
#     plt.imshow(image)
#     plt.show()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     plt.imshow(gray)
    
    # Perform text detection using pytesseract
    custom_config = r'--oem 3 --psm 6'  # OCR configuration
    text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)

    # Crop the regions where text is detected
    cropped_images = []
    for i, text in enumerate(text_data['text']):
        if text.strip():  # Check if the detected text is non-empty
#             print(text)
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            cropped_image = image[y:y+h, x:x+w]
#             plt.imshow(cropped_image)
#             plt.show()
            cropped_images.append(cropped_image)

    return cropped_images


def extractTextFromImage(inputImage):
    # getting results from the methods
    cropped_images = detect_and_crop_text(inputImage)
    li = []
    for cropped_image in cropped_images:
        li.append(displayResults(cropped_image))
    
    res = ''
    for i in li:
        res = res + " " + ''.join(i)
    return res
