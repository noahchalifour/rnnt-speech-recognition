from absl import logging
import json
import os
import re
import tensorflow as tf

try:
    from .utils.vocabulary import load_vocab
except ImportError:
    from utils.vocabulary import load_vocab

def load_model(path, checkpoint=None, verbose=True, expect_partial=False):

    model_config_filepath = os.path.join(path, 'config.json')
    vocab_filepath = os.path.join(path, 'vocab')
    checkpoints_path = os.path.join(path, 'checkpoints')

    vocab = load_vocab(vocab_filepath)

    with open(model_config_filepath, 'r') as model_config:
        model = Transducer.from_json(model_config.read())

    if checkpoint is None:
        _checkpoint = tf.train.latest_checkpoint(checkpoints_path)
    else:
        _checkpoint = os.path.join(checkpoints_path, checkpoint)

    if _checkpoint is None:

        if verbose:
            logging.info('Model restored without checkpoint.')

    else:

        if expect_partial:
            model.load_weights(_checkpoint).expect_partial()
        else:
            model.load_weights(_checkpoint)

        try:    
            model._checkpoint_step = int(re.findall(r'ckpt_(\d+)_', _checkpoint)[0])
        except Exception:
            if verbose:
                logging.warn('Could not determine checkpoint step, defaulting to 0.')

        if verbose:
            logging.info('Model restored from {}'.format(_checkpoint))

    return model, vocab


def encoder(num_layers,
            layer_size,
            tr_layer_index=None,
            tr_layer_factor=2):

    encoder_inp = tf.keras.layers.Input(shape=(None, 80))
    encoder_state_inp = tf.keras.layers.Input(shape=(1, None), batch_size=2)

    encoder_state = tf.unstack(encoder_state_inp)

    outputs = encoder_inp
    for i in range(num_layers):
        outputs, state_h, state_c = tf.keras.layers.LSTM(layer_size,
            return_sequences=True, return_state=True)(outputs, encoder_state)
        # if i == tr_layer_index:
        #     outputs = tf.keras.layers.Conv1D(layer_size, tr_layer_factor)(outputs)

    new_state = tf.stack([state_h, state_c])

    return tf.keras.Model(inputs=[encoder_inp, encoder_state_inp], outputs=[outputs, new_state])


def prediction_network(vocab_size,
                       embedding_size,
                       num_layers,
                       layer_size):

    inputs = tf.keras.layers.Input(shape=(None,))

    embed = tf.keras.layers.Embedding(vocab_size, embedding_size)(inputs)

    x = embed
    for _ in range(num_layers):
        x = tf.keras.layers.LSTM(layer_size,
            return_sequences=True)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def joint_network(num_units):

    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_units)
    ])


class Transducer(tf.keras.Model):

    def __init__(self,
                 vocab_size, 
                 embedding_size=64,
                 encoder_layers=8,
                 encoder_size=2048,
                 encoder_tr_index=1,
                 encoder_tr_factor=2,
                 pred_net_layers=2,
                 pred_net_size=2048,
                 joint_net_size=640,
                 softmax_size=4096):

        super(Transducer, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.encoder_size = encoder_size
        self.encoder_tr_index = encoder_tr_index
        self.encoder_tr_factor = encoder_tr_factor
        self.pred_net_layers = pred_net_layers
        self.pred_net_size = pred_net_size
        self.joint_net_size = joint_net_size
        self.softmax_size = softmax_size

        self._checkpoint_step = 0

        self._encoder = encoder(num_layers=self.encoder_layers, 
                                layer_size=self.encoder_size,
                                tr_layer_index=self.encoder_tr_index,
                                tr_layer_factor=self.encoder_tr_factor)

        self._pred_net = prediction_network(self.vocab_size, self.embedding_size,
            self.pred_net_layers, self.pred_net_size)
        
        self._joint_net = joint_network(self.joint_net_size)
        self._softmax = tf.keras.layers.Dense(self.softmax_size, activation='softmax')
        self._proj = tf.keras.layers.Dense(self.vocab_size)

    @classmethod
    def from_json(cls, json_str):

        return cls(**json.loads(json_str))

    def call(self, inputs, training=True):

        encoder_inp, pred_inp, encoder_state = inputs
        inputs_enc, new_enc_state = self._encoder([encoder_inp, encoder_state], 
            training=training)

        pred_outputs = self._pred_net(pred_inp, training=training)

        joint_inp = (tf.expand_dims(inputs_enc, 2)      # [B, T, V] => [B, T, 1, V]
            + tf.expand_dims(pred_outputs, 1))    # [B, U, V] => [B, 1, U, V]

        joint_out = self._joint_net(joint_inp)
        
        soft_out = self._softmax(joint_out)
        outputs = self._proj(soft_out)

        return outputs, new_enc_state

    def initial_state(self, batch_size):

        encoder_state = tf.stack([tf.zeros((batch_size, self.encoder_size)),
                                  tf.zeros((batch_size, self.encoder_size))])

        return encoder_state

    def to_json(self):

        return json.dumps({
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size,
            'encoder_layers': self.encoder_layers,
            'encoder_size': self.encoder_size,
            'pred_net_layers': self.pred_net_layers,
            'pred_net_size': self.pred_net_size,
            'joint_net_size': self.joint_net_size,
            'softmax_size': self.softmax_size
        })