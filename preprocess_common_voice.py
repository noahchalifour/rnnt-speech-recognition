from absl import app, logging, flags
import os
import json
import tensorflow as tf

from utils import preprocessing, encoding
from utils.data import common_voice
from hparams import *


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_dir', None,
    'Directory to read Common Voice data from.')
flags.DEFINE_string(
    'output_dir', './data',
    'Directory to save preprocessed data.')


def write_dataset(dataset, size, name):

    filepath_template = os.path.join(FLAGS.output_dir,
        '{}.tfrecord')

    writer = tf.data.experimental.TFRecordWriter(
        filepath_template.format(name))
    writer.write(dataset)

    with open(os.path.join(FLAGS.output_dir, 
        '{}-specs.json'.format(name)), 'w') as f:
        json.dump({
            'size': size
        }, f)


def main(_):

    hparams = {

        HP_TOKEN_TYPE: HP_TOKEN_TYPE.domain.values[1],
        HP_VOCAB_SIZE: HP_VOCAB_SIZE.domain.values[0],

        # Preprocessing
        HP_MEL_BINS: HP_MEL_BINS.domain.values[0],
        HP_FRAME_LENGTH: HP_FRAME_LENGTH.domain.values[0],
        HP_FRAME_STEP: HP_FRAME_STEP.domain.values[0],
        HP_HERTZ_LOW: HP_HERTZ_LOW.domain.values[0],
        HP_HERTZ_HIGH: HP_HERTZ_HIGH.domain.values[0]

    }

    _hparams = {k.name: v for k, v in hparams.items()}

    texts_gen = common_voice.texts_generator(FLAGS.data_dir)

    encoder_fn, decoder_fn, vocab_size = encoding.build_encoder(texts_gen,
        output_dir=FLAGS.output_dir, hparams=_hparams)
    _hparams[HP_VOCAB_SIZE.name] = vocab_size

    train_dataset, train_size = common_voice.load_dataset(
        FLAGS.data_dir, 'train')
    print('Train size:', train_size)
    train_dataset = preprocessing.preprocess_dataset(
        train_dataset,
        encoder_fn=encoder_fn,
        hparams=_hparams)
    write_dataset(train_dataset, train_size, 'train')

    dev_dataset, dev_size = common_voice.load_dataset(
        FLAGS.data_dir, 'dev')
    print('Dev size:', dev_size)
    dev_dataset = preprocessing.preprocess_dataset(
        dev_dataset,
        encoder_fn=encoder_fn,
        hparams=_hparams)
    write_dataset(dev_dataset, dev_size, 'dev')

    test_dataset, test_size = common_voice.load_dataset(
        FLAGS.data_dir, 'test')
    print('Test size:', test_size)
    test_dataset = preprocessing.preprocess_dataset(
        test_dataset,
        encoder_fn=encoder_fn,
        hparams=_hparams)
    write_dataset(test_dataset, test_size, 'test')


if __name__ == '__main__':

    flags.mark_flag_as_required('data_dir')

    app.run(main)