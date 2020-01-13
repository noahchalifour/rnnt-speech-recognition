from absl import logging
import json
import os
import re
import tensorflow as tf

try:
    from .utils.vocabulary import load_vocab
    from .utils.data.common import tf_mel_spectrograms
except ImportError:
    from utils.vocabulary import load_vocab
    from utils.data.common import tf_mel_spectrograms


def load_model(path, checkpoint=None, verbose=True, expect_partial=False):

    model_config_filepath = os.path.join(path, 'config.json')
    checkpoints_path = os.path.join(path, 'checkpoints')

    model = Transducer.load_json(model_config_filepath)

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

    return model


class Encoder(tf.keras.Model):

    def __init__(self,
                 num_layers,
                 d_model,
                 reduction_index=None,
                 reduction_factor=2):

        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.reduction_index = reduction_index
        self.reduction_factor = reduction_factor

        rnn_cell = lambda: tf.compat.v1.nn.rnn_cell.LSTMCell(self.d_model,
            num_proj=(self.d_model // 2))
        self.rnn_layers = [tf.keras.layers.RNN(rnn_cell(), return_sequences=True, return_state=True)
                           for _ in range(self.num_layers)]
        self.reduction_layer = tf.keras.layers.Conv1D(self.d_model // 2, self.reduction_factor)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, state):

        outputs = inputs

        next_state = tf.unstack(state)
        next_state[1] = next_state[1][:, :(self.d_model // 2)]

        for i in range(self.num_layers):

            # outputs, next_state = tf.compat.v1.nn.static_rnn(self.rnn_cell,
            #     outputs, initial_state=next_state)
            outputs, state_h, state_c = self.rnn_layers[i](outputs, next_state)
            outputs = self.layer_norm(outputs)

            next_state = [state_h, state_c]
            
            if i == self.reduction_index:
                outputs = self.reduction_layer(outputs)

        return outputs, next_state


class PredictionNetwork(tf.keras.Model):

    def __init__(self, 
                 vocab_size,
                 embedding_size,
                 num_layers,
                 layer_size):

        super(PredictionNetwork, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.layer_size = layer_size

        self.embed = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.rnn_layers = [tf.keras.layers.LSTM(self.layer_size, return_sequences=True)
                           for _ in range(self.num_layers)]
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):

        embedded = self.embed(inputs)
        
        outputs = embedded
        for i in range(self.num_layers):
            outputs = self.rnn_layers[i](outputs)
            outputs = self.layer_norm(outputs)

        return outputs


class Transducer(tf.keras.Model):

    def __init__(self,
                 vocab, 
                 embedding_size=64,
                 encoder_layers=8,
                 encoder_size=2048,
                 encoder_tr_index=1,
                 encoder_tr_factor=2,
                 pred_net_layers=2,
                 joint_net_size=640,
                 softmax_size=4096):

        super(Transducer, self).__init__()

        self.vocab_size = len(vocab)
        self.vocab = vocab
        
        self._vocab_t = tf.constant(vocab)
        self._vocab_hash = tf.reduce_sum(tf.strings.unicode_decode(self._vocab_t, 'UTF-8'), axis=1)

        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.encoder_size = encoder_size
        self.encoder_tr_index = encoder_tr_index
        self.encoder_tr_factor = encoder_tr_factor
        self.pred_net_layers = pred_net_layers
        self.pred_net_size = encoder_size // 2
        self.joint_net_size = joint_net_size
        self.softmax_size = softmax_size

        self._checkpoint_step = 0

        self.encoder = Encoder(num_layers=self.encoder_layers,
                               d_model=self.encoder_size,
                               reduction_index=self.encoder_tr_index,
                               reduction_factor=self.encoder_tr_factor)

        self.prediction_network = PredictionNetwork(vocab_size=self.vocab_size,
                                                    embedding_size=self.embedding_size,
                                                    num_layers=self.pred_net_layers,
                                                    layer_size=self.pred_net_size)
        
        self._joint_net = tf.keras.layers.Dense(self.joint_net_size)
        self._softmax = tf.keras.layers.Dense(self.softmax_size, activation='softmax')
        self._proj = tf.keras.layers.Dense(self.vocab_size)

    @classmethod
    def load_json(cls, filepath):

        with open(filepath, 'r') as f:
            json_dict = json.loads(f.read())

        json_dict['vocab'] = load_vocab(os.path.join(os.path.dirname(filepath), json_dict['vocab']))

        return cls(**json_dict)

    def call(self, inputs, training=True):

        encoder_inp, pred_inp, encoder_state = inputs

        inputs_enc, new_enc_state = self.encoder(encoder_inp, encoder_state, 
            training=training)

        pred_outputs = self.prediction_network(pred_inp)

        joint_inp = (tf.expand_dims(inputs_enc, 2)      # [B, T, V] => [B, T, 1, V]
            + tf.expand_dims(pred_outputs, 1))    # [B, U, V] => [B, 1, U, V]

        joint_out = self._joint_net(joint_inp)
        
        soft_out = self._softmax(joint_out)
        outputs = self._proj(soft_out)

        return outputs, new_enc_state

    @tf.function
    def predict(self, audio, sr, pred_inp, enc_state):

        # NOTE: Can only run predict of first input

        _audio = audio[0]
        _sr = sr[0]
        _enc_state = enc_state[0]

        pred_inp_c = tf.strings.bytes_split(pred_inp).flat_values
        pred_inp_uni = tf.strings.unicode_decode(pred_inp_c, 'UTF-8').flat_values
        pred_inp_r = tf.expand_dims(pred_inp_uni, axis=-1)

        pred_inp_enc = tf.concat([[0],
                                 tf.where(tf.equal(pred_inp_r, self._vocab_hash))[:, -1]],
            axis=0)
        pred_inp_enc = tf.expand_dims(pred_inp_enc, axis=0)

        specs = tf_mel_spectrograms(_audio, _sr)
        expanded_specs = tf.expand_dims(specs, axis=0)

        expanded_specs.set_shape([1, None, 80])

        predictions, new_enc_state = self.call([expanded_specs, pred_inp_enc, _enc_state], 
            training=False)
        predictions = predictions[:, -1, -1, :]
        predictions = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        pred_hash = [tf.expand_dims(self._vocab_hash[predictions[0]], axis=0)]
        pred_char = tf.strings.unicode_encode(pred_hash, 'UTF-8')

        return pred_char, new_enc_state[0], new_enc_state[1]

    def initial_state(self, batch_size):

        encoder_state = [tf.zeros((batch_size, self.encoder_size)),
                         tf.zeros((batch_size, self.encoder_size))]

        return encoder_state

    def to_json(self):

        return json.dumps({
            'vocab': './vocab',
            'embedding_size': self.embedding_size,
            'encoder_layers': self.encoder_layers,
            'encoder_size': self.encoder_size,
            'encoder_tr_index': self.encoder_tr_index,
            'encoder_tr_factor': self.encoder_tr_factor,
            'pred_net_layers': self.pred_net_layers,
            'joint_net_size': self.joint_net_size,
            'softmax_size': self.softmax_size
        })

    def save_config(self, filepath):

        with open(filepath, 'w') as model_config:
            model_config.write(self.to_json())