# Arda Mavi
import os
import numpy as np
from os import listdir
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize, toimage, imsave
from sklearn.model_selection import train_test_split

def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path)
    img = imresize(img, (64, 64, 3))
    return img

def save_img(img, name='segmentated.jpg'):
    imsave(name, img.reshape(64, 64, 3))

def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    try:
        X = np.load('Data/npy_train_data/X.npy')
        Y = np.load('Data/npy_train_data/Y.npy')
    except:
        inputs_path = dataset_path+'/input'
        images = listdir(inputs_path) # Geting images
        X = []
        Y = []
        for img in images:
            img_path = inputs_path+'/'+img
            x_img = get_img(img_path)
            x_img = x_img.astype('float32')
            y_img = get_img(img_path.replace('input', 'output'))
            y_img = y_img.astype('float32')

            mean = np.mean(x_img)
            std = np.std(x_img)
            x_img -= mean
            x_img /= std
            x_img /= 255.

            mean = np.mean(y_img)
            std = np.std(y_img)
            y_img -= mean
            y_img /= std
            y_img /= 255.

            X.append(x_img)
            Y.append(y_img)
        X = np.array(X)
        Y = np.array(Y)
        # Create dateset:
        if not os.path.exists('Data/npy_train_data/'):
            os.makedirs('Data/npy_train_data/')
        np.save('Data/npy_train_data/X.npy', X)
        np.save('Data/npy_train_data/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test
