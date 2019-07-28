from flask import Flask, render_template,url_for, request, jsonify
import numpy as np
import tensorflow as tf
import re
import io
import base64

import time
import pickle
import matplotlib.pyplot as plt
import keras
from keras import backend as K
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, Reshape
# from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
# from keras.layers import LeakyReLU, Dropout
# from keras.layers import BatchNormalization
# from keras.optimizers import Adam, RMSprop

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_generator():
    with open('GAN-models/generator_model (1).pkl', 'rb') as handle:
        generator_model = pickle.load(handle)
    return generator_model

def load_discriminator():
    with open('GAN-models/discriminator_model (1).pkl', 'rb') as handle:
        discriminator_model = pickle.load(handle)
    return discriminator_model

def load_reality_check():
    with open('GAN-models/mnist_model.pkl', 'rb') as handle:
        mnist_model = pickle.load(handle)
    return mnist_model

def generate_num():
    K.clear_session()

    generator_model = load_generator()
    discriminator_model = load_discriminator()
    mnist_model = load_reality_check()

    probability = np.array([np.zeros(10)])
    a = np.array([np.zeros(1)])
    n_tries = 0
    prob = 0
    save_prob = 0
    save_num = 0
    save_prob = 0
    # try 2000 times until we get to 0.9999 confidence level
    for _ in range(1000):
        if save_prob < 0.99999:
            n_tries = n_tries+1
            noise_plot = np.random.uniform(-1.0, 1.0, size=[1,100])

            generator_model._make_predict_function()
            discriminator_model._make_predict_function()
            mnist_model._make_predict_function()

            num = generator_model.predict(noise_plot)
            a = discriminator_model.predict(num)
            probability_all = mnist_model.predict(num)
            prob = probability_all[0][4]
        # in case we find a good number in the first try
        if n_tries == 1:
            save_prob = prob
            save_num = num
            save_a = a
            save_probability_all = probability_all
        # save the best number
        if prob > save_prob:
            save_prob = prob
            save_num = num
            save_a = a
            save_probability_all = probability_all

    img = io.BytesIO()
    plt.imshow(save_num.reshape(28,28),cmap='binary')
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    K.clear_session()
    return 'data:image/png;base64,{}'.format(graph_url), save_prob, n_tries
