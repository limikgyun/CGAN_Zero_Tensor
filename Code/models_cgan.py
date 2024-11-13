import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Embedding, Concatenate, Conv2D, Conv2DTranspose, LeakyReLU, ReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class CustomActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
    def call(self, inputs):
        logexpsum = K.sum(K.exp(inputs), axis=-1, keepdims=True)
        result = logexpsum / (logexpsum + 1.0)
        return result
    
def define_discriminator_c(n_classes, in_shape=(1, 256, 1)):
    # 이미지 입력
    in_image = Input(shape=in_shape)
    # 레이블 입력
    in_label = Input(shape=(1,))
    # 레이블 임베딩
    li = Embedding(n_classes, 256)(in_label)
    # # 레이블을 7x7 크기로 변환
    # n_nodes = in_shape[0] * in_shape[1]
    li = Dense(256)(li)
    li = Reshape((1, 256, 1))(li)
    # 이미지와 레이블 병합
    # print(in_image.shape, li.shape)
    merged = Concatenate()([in_image, li])

    # # 합성된 입력을 판별
    # fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(merged)
    # fe = LeakyReLU(alpha=0.2)(fe)
    # fe = Dropout(0.4)(fe)
    # fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(fe)
    # fe = LeakyReLU(alpha=0.2)(fe)
    # fe = Dropout(0.4)(fe)
    # fe = Flatten()(fe)

    fe = Conv2D(filters=32, kernel_size=(1, 5))(merged)
    fe = LeakyReLU()(fe)
    # fe = Dropout(0.4)(fe)
    fe = Conv2D(filters=32, kernel_size=(1, 5))(fe)
    fe = LeakyReLU()(fe)
    # fe = Dropout(0.4)(fe)
    fe = Conv2D(filters=32, kernel_size=(1, 5))(fe)
    fe = LeakyReLU()(fe)
    # fe = Dropout(0.4)(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    # fe = Dense(1)(fe)
    # d_out_layer = CustomActivation()(fe)
    # d_out_layer = Dense(1, activation=CustomActivation())(fe)
    # print(d_out_layer.shape)
    fe = CustomActivation()(fe)
    d_out_layer = Dense(1, activation='sigmoid')(fe)
    d_model = Model([in_image, in_label], d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

    # out_layer = Dense(1, activation='sigmoid')(fe)
    # # 모델 정의
    # model = Model([in_image, in_label], out_layer)
    # opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return d_model

def define_generator_c(n_classes):
    # 잠재 공간 입력
    in_lat = Input(shape=(512,))
    # 레이블 입력
    in_label = Input(shape=(1,))
    # 레이블 임베딩
    li = Flatten()(Embedding(n_classes, 512)(in_label))
    # 레이블을 잠재 공간 크기로 변환
    li = Dense(512)(li)
    # 잠재 공간과 레이블 병합
    merged = Concatenate()([in_lat, li])
    
    n_nodes = 1 * 256 * 32
    gen = Dense(n_nodes)(merged)
    gen = ReLU()(gen)
    gen = Reshape((1, 256, 32))(gen)

    gen = Conv2D(filters = 32, kernel_size=(1,20), strides=1)(gen)
    gen = ReLU()(gen)   
    gen = Conv2D(filters = 32, kernel_size=(1,40), strides=1)(gen)
    gen = ReLU()(gen)
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,40), strides=1)(gen)
    gen = ReLU()(gen)
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,20), strides=1)(gen)
    gen = ReLU()(gen)

    out_layer = Conv2DTranspose(filters = 1, kernel_size= (1, 1), strides=1, activation='tanh', padding='same')(gen)
    g_model = Model([in_lat, in_label], out_layer)  # 두 입력을 모두 포함
    return g_model

def define_gan_c(g_model, d_model):
    # 판별기 가중치 고정
    d_model.trainable = False
    # GAN 입력
    gen_noise, gen_label = g_model.input
    # 생성기 출력
    gen_output = g_model.output
    # 판별기 입력
    gan_output = d_model([gen_output, gen_label])
    # GAN 모델 정의
    gan_model = Model([gen_noise, gen_label], gan_output)
    # # 컴파일
    # opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # gan_model.compile(loss='binary_crossentropy', optimizer=opt)
    gan_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return gan_model

if __name__ == '__main__':    
    # 모델 생성
    # latent_dim = 1024
    n_classes = 10
    d_model = define_discriminator_c(n_classes)
    g_model = define_generator_c(n_classes)
    gan_model = define_gan_c(g_model, d_model)

    # 모델 요약 출력
    d_model.summary()
    g_model.summary()
    gan_model.summary()