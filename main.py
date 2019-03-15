import numpy
import argparse

import tensorflow as tf
import tensorflow.contrib.layers.variance_scaling_initializer as he_initializer

import keras
from keras.applications.densenet import DenseNet201
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda, multiply
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K


from autoInt import AutoInt


def nn_layer(inputs, inputs_size, n_hidden, add_bias=True, name=None, activation=None):
    weights_size = [inputs_size, n_hidden]
    layer_weights = tf.Variable(he_initializer()(shape=weights_size), name=name + 'weigths', trainable=True)

    hidden_layer = tf.matmul(inputs, layer_weights)

    if add_bias == True:
        layer_bias = tf.Variable(tf.zeros([n_hidden]), name=name + 'bias', trainable=True)
        hidden_layer = tf.add(hidden_layer, layer_bias)

    if activation != None:
        hidden_layer = activation(hidden_layer)

    return hidden_layer


if __name__ == '__main__':
    config = None
    feature_size=None
    num_embedding=None
    embedding_dim=None
    output_dim = None
    n_class = None

    """ Pretrained model using Keras """
    # To extract feature of image data, use DenseNet201
    cnn_model = DenseNet201(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

    # Image feature
    feature = cnn_model.output  # Feature Extraction
    gap = GlobalAveragePooling2D()(feature)  # (n_channel,)

    cnn_model = Model(inputs=cnn_model.input, outputs=gap)
    cnn_model_n_channel = cnn_model.layers[-1].output_shape[-1]


    """ Tensorflow """
    Y = tf.placeholder(tf.int8, shape=[None, n_class], name="label")

    # Layer for image feature vector
    img_feature_vector = tf.placeholder(tf.float32, shape=[None, cnn_model_n_channel], name="img_feature_vector")
    img_output_layer = nn_layer(img_feature_vector, inputs_size=cnn_model_n_channel, n_hidden=output_dim, activation=tf.nn.relu, name="img_output_layer")


    # To extract feature of context data(text, tabular), use AutoInt
    autoInt = AutoInt(feature_size=feature_size,
                            num_embedding=num_embedding,
                            embedding_dim=embedding_dim,
                            output_dim=output_dim)

    autoInt_layer = autoInt.output

    # multiply attention(autoInt) to img_output_layer
    output = tf.mutiply(img_output_layer, autoInt_layer)

    # classifier
    classifier_weight = tf.Variable(he_initializer()(shape=[output_dim, n_class]), name='classifier_weights')
    classifier_bias = tf.Variable(tf.zeros([n_class]), name='classifier_bias')
    classifier_layer = tf.matmul(output, classifier_weight) + classifier_bias

    # probability
    softmax_layer = tf.nn.softmax(classifier_layer)


    # Loss
    learning_rate = config.learning_rate
    beta_1 = config.beta_1
    beta_2 = config.beta_2
    epsilon = config.epsilon

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classifier_layer, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta_1, beta2=beta_2,
                                       epsilon=epsilon).minimize(cost)
