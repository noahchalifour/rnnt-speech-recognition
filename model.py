import re
import os
import tensorflow as tf

from hparams import *


class TimeReduction(tf.keras.layers.Layer):

    def __init__(self, 
                 reduction_factor,
                 batch_size=None,
                 **kwargs):

        super(TimeReduction, self).__init__(**kwargs)

        self.reduction_factor = reduction_factor
        self.batch_size = batch_size

    def call(self, inputs):

        input_shape = tf.shape(inputs)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = input_shape[0]

        max_time = input_shape[1]
        num_units = inputs.get_shape().as_list()[-1]

        outputs = inputs

        paddings = [[0, 0], [0, tf.math.floormod(max_time, self.reduction_factor)], [0, 0]]
        outputs = tf.pad(outputs, paddings)

        return tf.reshape(outputs, (batch_size, -1, num_units * self.reduction_factor))


def encoder(specs_shape,
            num_layers,
            d_model,
            proj_size,
            reduction_index,
            reduction_factor,
            stateful=False,
            initializer=None,
            dtype=tf.float32):

    batch_size = None
    if stateful:
        batch_size = 1

    mel_specs = tf.keras.Input(shape=specs_shape, batch_size=batch_size, 
        dtype=tf.float32)

    lstm_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(d_model, 
        num_proj=proj_size, initializer=initializer, dtype=dtype)

    outputs = mel_specs

    for i in range(num_layers):

        rnn_layer = tf.keras.layers.RNN(lstm_cell(), 
            return_sequences=True, stateful=stateful)

        outputs = rnn_layer(outputs)
        outputs = tf.keras.layers.LayerNormalization(dtype=dtype)(outputs)

        if i == reduction_index:
            outputs = TimeReduction(reduction_factor,
                batch_size=batch_size)(outputs)

    return tf.keras.Model(inputs=[mel_specs], outputs=[outputs], 
        name='encoder')


def prediction_network(vocab_size,
                       embedding_size,
                       num_layers,
                       layer_size,
                       proj_size,
                       stateful=False,
                       initializer=None,
                       dtype=tf.float32):

    batch_size = None
    if stateful:
        batch_size = 1

    inputs = tf.keras.Input(shape=[None], batch_size=batch_size,
        dtype=tf.int32)

    embed = tf.keras.layers.Embedding(vocab_size, embedding_size)(inputs)

    rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(layer_size,
        num_proj=proj_size, initializer=initializer, dtype=dtype)
    
    outputs = embed

    for _ in range(num_layers):

        outputs = tf.keras.layers.RNN(rnn_cell(), 
            return_sequences=True)(outputs)
        outputs = tf.keras.layers.LayerNormalization(dtype=dtype)(outputs)

    return tf.keras.Model(inputs=[inputs], outputs=[outputs],
        name='prediction_network')


def build_keras_model(hparams,
                      stateful=False,
                      initializer=None,
                      dtype=tf.float32):

    specs_shape = [None, hparams[HP_MEL_BINS.name]]

    batch_size = None
    if stateful:
        batch_size = 1

    mel_specs = tf.keras.Input(shape=specs_shape, batch_size=batch_size,
        dtype=tf.float32, name='mel_specs')
    pred_inp = tf.keras.Input(shape=[None], batch_size=batch_size,
        dtype=tf.float32, name='pred_inp')

    inp_enc = encoder(
        specs_shape=specs_shape,
        num_layers=hparams[HP_ENCODER_LAYERS.name],
        d_model=hparams[HP_ENCODER_SIZE.name],
        proj_size=hparams[HP_PROJECTION_SIZE.name],
        reduction_index=hparams[HP_TIME_REDUCT_INDEX.name],
        reduction_factor=hparams[HP_TIME_REDUCT_FACTOR.name],
        stateful=stateful,
        initializer=initializer,
        dtype=dtype)(mel_specs)

    pred_outputs = prediction_network(
        vocab_size=hparams[HP_VOCAB_SIZE.name],
        embedding_size=hparams[HP_EMBEDDING_SIZE.name],
        num_layers=hparams[HP_PRED_NET_LAYERS.name],
        layer_size=hparams[HP_PRED_NET_SIZE.name],
        proj_size=hparams[HP_PROJECTION_SIZE.name],
        stateful=stateful,
        initializer=initializer,
        dtype=dtype)(pred_inp)

    inp_enc_exp = tf.expand_dims(inp_enc, axis=2)               # [B, T, V] => [B, T, 1, V]
    pred_outputs_exp = tf.expand_dims(pred_outputs, axis=1)     # [B, U, V] => [B, 1, U, V]

    # inp_enc_exp = tf.keras.layers.Reshape(
    #     (-1, 1, hparams[HP_PROJECTION_SIZE.name]))(inp_enc)        
    # pred_outputs_exp = tf.keras.layers.Reshape(
    #     (1, -1, hparams[HP_PROJECTION_SIZE.name]))(pred_outputs)   

    inp_enc_b = tf.tile(inp_enc_exp, tf.stack([1, 1, tf.shape(pred_outputs)[1], 1], 
        name='stack_inp_enc'))
    pred_out_b = tf.tile(pred_outputs_exp, tf.stack([1, tf.shape(inp_enc)[1], 1, 1], 
        name='stack_pred_out'))

    joint_inp = tf.keras.layers.concatenate([inp_enc_b, pred_out_b])

    joint_outputs = tf.keras.layers.Dense(hparams[HP_JOINT_NET_SIZE.name],
        kernel_initializer=initializer, activation='tanh')(joint_inp)

    outputs = tf.keras.layers.Dense(hparams[HP_VOCAB_SIZE.name],
        kernel_initializer=initializer)(joint_outputs)

    return tf.keras.Model(inputs=[mel_specs, pred_inp],
        outputs=[outputs], name='transducer')

