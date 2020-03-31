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


