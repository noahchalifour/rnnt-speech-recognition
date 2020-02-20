from absl import logging
import os
import json
import re

from model import build_keras_model


def load_hparams(model_dir):

    with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
        return json.load(f)


def save_hparams(hparams, model_dir):

    with open(os.path.join(model_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f)


def load_model(model_dir, vocab_size, hparams, training=True):

    model, loss_fn = build_keras_model(vocab_size, hparams,
        training=training)
    
    epochs_r = re.compile(r'model.\d+-')
    newest_weights = max(filter(epochs_r.findall, os.listdir(model_dir)))

    model.load_weights(os.path.join(model_dir, newest_weights))

    logging.info('Restored weights from {}.'.format(newest_weights))

    return model, loss_fn