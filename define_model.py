# coding: utf-8
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import concatenate
import numpy as np
import os
from keras.optimizers import Adam
import cv2
import h5py
from keras import backend as K

def reflection_padding(x, padding):
    reflected = Lambda(lambda x: x[:, :, ::-1, :])(x)
    reflected = Lambda(lambda x: x[:, :, :padding[1], :])(reflected)
    upper_row = concatenate([x, reflected], axis=2)
    lower_row = Lambda(lambda x: x[:, ::-1, :, :])(upper_row)
    lower_row = Lambda(lambda x: x[:, :padding[0], :, :])(lower_row)
    padded = concatenate([upper_row, lower_row], axis=1)
    return padded

def conv_bn_relu(x, size, filters, kernel_size, strides):
    padding = [0, 0]
    padding[0] =  (int(size[0]/strides[0]) - 1) * strides[0] + kernel_size - size[0]
    padding[1] =  (int(size[1]/strides[1]) - 1) * strides[1] + kernel_size - size[1]
    x = reflection_padding(x, padding)

    x = Conv2D(filters, kernel_size, strides=strides)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    new_size = [int(size[0]/strides[0]), int(size[1]/strides[1])]
    return x, new_size

def down_sampling(x, size, filters, kernel_size):
    new_size = [size[0], size[1]]
    if size[0] % 2 != 0:
        x = reflection_padding(x, [1, 0])
        new_size[0] = size[0] + 1
    if size[1] % 2 != 0:
        x = reflection_padding(x, [0, 1])
        new_size[1] = size[1] + 1
    size = new_size
    x, size = conv_bn_relu(x, size, filters, kernel_size, (2, 2))
    x, size = conv_bn_relu(x, size, filters, kernel_size, (1, 1))
    return x, size

def upsample(x, size, inter):
    x = UpSampling2D(size=(2, 2))(x)
    if inter == "bilinear":
        x_padded = reflection_padding(x, (1, 1))
        x = Lambda(lambda x: (x[:, :-1, 1:, :] + x[:, 1:, :-1, :] + x[:, :-1, :-1, :] + x[:, :-1, :-1, :]) / 4.0)(x_padded)
    return x, [size[0]*2, size[1]*2]

def up_sampling(x, size, filters, kernel_size, inter):
    x, size = upsample(x, size, inter)
    x, size = conv_bn_relu(x, size, filters, kernel_size, (1, 1))
    x, size = conv_bn_relu(x, size, filters, 1, (1, 1))
    return x, size

def skip(x, size, filters, kernel_size):
    x, size = conv_bn_relu(x, size, filters, kernel_size, (1, 1))
    return x, size

def define_model(num_u, num_d, kernel_u, kernel_d, num_s, kernel_s, height, width, inter, lr, input_channel=32):
    depth = len(num_u)
    size = [height, width]

    inputs = Input(shape=(height, width, input_channel))

    x = inputs
    down_sampled = []
    sizes = [size]
    for i in range(depth):
        x, size = down_sampling(x, size, num_d[i], kernel_d[i])
        down_sampled.append(x)
        sizes.append(size)

    for i in range(depth-1, -1, -1):
        if num_s[i] != 0:
            skipped, size = skip(down_sampled[i], size, num_s[i], kernel_s[i])
            x = concatenate([x, skipped], axis=3)
        x, size = up_sampling(x, size, num_u[i], kernel_u[i], inter)

        if sizes[i] != size:
            x = Lambda(lambda x: x[:, :sizes[i][0], :sizes[i][1], :])(x)
            size = sizes[i]

    x = Conv2D(3, 1)(x)
    model = Model(inputs, x)

    return model
