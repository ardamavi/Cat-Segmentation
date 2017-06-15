# Arda Mavi
import os
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Conv2DTranspose, concatenate

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def save_model(model):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    model_json = model.to_json()
    with open("Data/Model/model.json", "w") as model_file:
        model_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Data/Model/weights.h5")
    print('Model and weights saved')
    return

def get_model():

    inputs = Input(shape=(64, 64, 3))

    conv_1 = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(act_1)
    act_2 = Activation('relu')(conv_2)

    deconv_1 = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same')(act_2)
    act_3 = Activation('relu')(deconv_1)

    merge_1 = concatenate([act_3, act_1], axis=3)

    deconv_2 = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same')(merge_1)
    act_4 = Activation('relu')(deconv_2)

    model = Model(inputs=[inputs], outputs=[act_4])

    model.compile(optimizer='adadelta', loss=dice_coef_loss, metrics=[dice_coef])

    return model

if __name__ == '__main__':
    save_model(get_model())
