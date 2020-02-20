import re
import os
import tensorflow as tf

from utils.loss import get_loss_fn
from hparams import *


def encoder(specs_shape,
            num_layers,
            d_model,
            reduction_index,
            reduction_factor,
            stateful=False):

    batch_size = None
    if stateful:
        batch_size = 1

    mel_specs = tf.keras.Input(shape=specs_shape, batch_size=batch_size, 
        dtype=tf.float32)

    rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(d_model,
        num_proj=(d_model // 2))

    outputs = mel_specs

    for i in range(num_layers):

        rnn_layer = tf.keras.layers.RNN(rnn_cell(), 
            return_sequences=True, stateful=stateful)
        outputs = rnn_layer(outputs)
        outputs = tf.keras.layers.LayerNormalization()(outputs)

        if i == reduction_index:
            outputs = tf.keras.layers.Conv1D(d_model // 2, 
                reduction_factor)(outputs)

    return tf.keras.Model(inputs=[mel_specs], 
        outputs=[outputs])


def prediction_network(vocab_size,
                       embedding_size,
                       num_layers,
                       layer_size):

    inputs = tf.keras.Input(shape=[None], dtype=tf.int32)

    embed = tf.keras.layers.Embedding(vocab_size, 
        embedding_size)(inputs)
    
    outputs = embed

    for _ in range(num_layers):

        outputs = tf.keras.layers.LSTM(layer_size, 
            return_sequences=True)(outputs)
        outputs = tf.keras.layers.LayerNormalization()(outputs)

    return tf.keras.Model(inputs=[inputs], outputs=[outputs])


def build_keras_model(vocab_size,
                      hparams,
                      training=True):

    specs_shape = [None, hparams[HP_MEL_BINS.name]]

    batch_size = None
    if not training:
        batch_size = 1

    mel_specs = tf.keras.Input(shape=specs_shape, batch_size=batch_size,
        dtype=tf.float32, name='mel_specs')
    pred_inp = tf.keras.Input(shape=[None], batch_size=batch_size,
        dtype=tf.int32, name='pred_inp')
    spec_lengths = tf.keras.Input(shape=[], batch_size=batch_size,
        dtype=tf.int32, name='spec_lengths')
    label_lengths = tf.keras.Input(shape=[], batch_size=batch_size,
        dtype=tf.int32, name='label_lengths')

    stateful_rnn = training == False

    inp_enc = encoder(
        specs_shape=specs_shape,
        num_layers=hparams[HP_ENCODER_LAYERS.name],
        d_model=hparams[HP_ENCODER_SIZE.name],
        reduction_index=hparams[HP_TIME_REDUCT_INDEX.name],
        reduction_factor=hparams[HP_TIME_REDUCT_FACTOR.name],
        stateful=stateful_rnn)(mel_specs)

    pred_net_size = hparams[HP_ENCODER_SIZE.name] // 2

    pred_outputs = prediction_network(
        vocab_size=vocab_size,
        embedding_size=hparams[HP_EMBEDDING_SIZE.name],
        num_layers=hparams[HP_PRED_NET_LAYERS.name],
        layer_size=pred_net_size)(pred_inp)

    joint_inp = (tf.expand_dims(inp_enc, 2)      # [B, T, V] => [B, T, 1, V]
        + tf.expand_dims(pred_outputs, 1))       # [B, U, V] => [B, 1, U, V]
    joint_outputs = tf.keras.layers.Dense(hparams[HP_JOINT_NET_SIZE.name])(joint_inp)

    soft_outputs = tf.keras.layers.Dense(hparams[HP_SOFTMAX_SIZE.name], 
        activation='softmax')(joint_outputs)

    outputs = tf.keras.layers.Dense(vocab_size)(soft_outputs)

    loss_fn = get_loss_fn(spec_lengths, label_lengths)

    return tf.keras.Model(inputs=[mel_specs, pred_inp, spec_lengths, label_lengths],
        outputs=[outputs]), loss_fn