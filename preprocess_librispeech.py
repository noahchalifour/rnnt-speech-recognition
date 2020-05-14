from absl import app, logging, flags
import os
import json
import tensorflow as tf

from utils import preprocessing, encoding
from utils.data import librispeech
from hparams import *


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_dir', None,
    'Directory to read Librispeech data from.')
flags.DEFINE_string(
    'output_dir', './data',
    'Directory to save preprocessed data.')
flags.DEFINE_integer(
    'max_length', 0,
    'Max audio length in seconds.')


def write_dataset(dataset, name):

    filepath = os.path.join(FLAGS.output_dir,
        '{}.tfrecord'.format(name))

    writer = tf.data.experimental.TFRecordWriter(filepath)
    writer.write(dataset)

    logging.info('Wrote {} dataset to {}'.format(
        name, filepath))


def main(_):

    hparams = {

        HP_TOKEN_TYPE: HP_TOKEN_TYPE.domain.values[1],
        HP_VOCAB_SIZE: HP_VOCAB_SIZE.domain.values[0],

        # Preprocessing
        HP_MEL_BINS: HP_MEL_BINS.domain.values[0],
        HP_FRAME_LENGTH: HP_FRAME_LENGTH.domain.values[0],
        HP_FRAME_STEP: HP_FRAME_STEP.domain.values[0],
        HP_HERTZ_LOW: HP_HERTZ_LOW.domain.values[0],
        HP_HERTZ_HIGH: HP_HERTZ_HIGH.domain.values[0],
        HP_DOWNSAMPLE_FACTOR: HP_DOWNSAMPLE_FACTOR.domain.values[0]

    }

    train_splits = [
        'dev-clean'
    ]

    dev_splits = [
        'dev-clean'
    ]

    test_splits = [
        'dev-clean'
    ]

    # train_splits = [
    #     'train-clean-100',
    #     'train-clean-360',
    #     'train-other-500'
    # ]

    # dev_splits = [
    #     'dev-clean',
    #     'dev-other'
    # ]

    # test_splits = [
    #     'test-clean',
    #     'test-other'
    # ]

    _hparams = {k.name: v for k, v in hparams.items()}

    texts_gen = librispeech.texts_generator(FLAGS.data_dir,
        split_names=train_splits)

    encoder_fn, decoder_fn, vocab_size = encoding.get_encoder(
        output_dir=FLAGS.output_dir,
        hparams=_hparams,
        texts_generator=texts_gen)
    _hparams[HP_VOCAB_SIZE.name] = vocab_size

    train_dataset = librispeech.load_dataset(
        FLAGS.data_dir, train_splits)
    dev_dataset = librispeech.load_dataset(
        FLAGS.data_dir, dev_splits)
    test_dataset = librispeech.load_dataset(
        FLAGS.data_dir, test_splits)

    train_dataset = preprocessing.preprocess_dataset(
        train_dataset,
        encoder_fn=encoder_fn,
        hparams=_hparams,
        max_length=FLAGS.max_length,
        save_plots=True)
    write_dataset(train_dataset, 'train')

    dev_dataset = preprocessing.preprocess_dataset(
        dev_dataset,
        encoder_fn=encoder_fn,
        hparams=_hparams,
        max_length=FLAGS.max_length)
    write_dataset(dev_dataset, 'dev')

    test_dataset = preprocessing.preprocess_dataset(
        test_dataset,
        encoder_fn=encoder_fn,
        hparams=_hparams,
        max_length=FLAGS.max_length)
    write_dataset(test_dataset, 'test')


if __name__ == '__main__':

    flags.mark_flag_as_required('data_dir')

    app.run(main)
