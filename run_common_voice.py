from absl import flags, logging, app
from tensorboard.plugins.hparams import api as hp
import json
import os
import tensorflow as tf

from utils.data import common_voice
from utils import preprocessing, vocabulary
from model import build_keras_model
from hparams import *

FLAGS = flags.FLAGS

# Required flags
flags.DEFINE_enum(
    'mode', None,
    ['train'],
    'Mode to run.')
flags.DEFINE_string(
    'data_dir', None,
    'Input data directory.')

# Optional flags
flags.DEFINE_string(
    'tb_log_dir', './logs',
    'Directory to save Tensorboard logs.')
flags.DEFINE_string(
    'model_dir', './model',
    'Directory to save model.')
flags.DEFINE_integer(
    'batch_size', 32,
    'Training batch size.')
flags.DEFINE_integer(
    'n_epochs', 1000,
    'Number of training epochs.')


def get_dataset_fn(base_path, 
                   vocab_table, 
                   batch_size, 
                   hparams):

    def _dataset_fn(name):

        dataset, dataset_size = common_voice.load_dataset(base_path, name)

        dataset = preprocessing.preprocess_dataset(dataset, 
            vocab_table=vocab_table,
            batch_size=batch_size,
            hparams=hparams)

        steps_per_epoch = dataset_size // batch_size

        return dataset, steps_per_epoch

    return _dataset_fn


def train():

    hparams = {

        # Preprocessing
        HP_MEL_BINS: HP_MEL_BINS.domain.values[0],
        HP_FRAME_LENGTH: HP_FRAME_LENGTH.domain.values[0],
        HP_FRAME_STEP: HP_FRAME_STEP.domain.values[0],
        HP_HERTZ_LOW: HP_HERTZ_LOW.domain.values[0],
        HP_HERTZ_HIGH: HP_HERTZ_HIGH.domain.values[0],

        # Model
        HP_EMBEDDING_SIZE: HP_EMBEDDING_SIZE.domain.values[0],
        HP_ENCODER_LAYERS: HP_ENCODER_LAYERS.domain.values[0],
        HP_ENCODER_SIZE: HP_ENCODER_SIZE.domain.values[0],
        HP_TIME_REDUCT_INDEX: HP_TIME_REDUCT_INDEX.domain.values[0],
        HP_TIME_REDUCT_FACTOR: HP_TIME_REDUCT_FACTOR.domain.values[0],
        HP_PRED_NET_LAYERS: HP_PRED_NET_LAYERS.domain.values[0],
        HP_JOINT_NET_SIZE: HP_JOINT_NET_SIZE.domain.values[0],
        HP_SOFTMAX_SIZE: HP_SOFTMAX_SIZE.domain.values[0],

        HP_LEARNING_RATE: HP_LEARNING_RATE.domain.values[0]

    }

    vocab = vocabulary.init_vocab()
    vocab_table = preprocessing.build_lookup_table(vocab, 
        default_value=0)

    vocab_size = len(vocab)

    logging.info('Vocabulary size: {}'.format(vocab_size))

    dataset_fn = get_dataset_fn(FLAGS.data_dir, 
        vocab_table=vocab_table,
        batch_size=FLAGS.batch_size,
        hparams=hparams)

    train_dataset, train_steps = dataset_fn('train')
    dev_dataset, dev_steps = dataset_fn('dev')

    model, loss_fn = build_keras_model(vocab_size, hparams)
    optimizer = tf.keras.optimizers.Adam(hparams[HP_LEARNING_RATE])

    model.compile(loss=loss_fn, optimizer=optimizer, 
        experimental_run_tf_function=False)

    os.makedirs(FLAGS.model_dir, exist_ok=True)
    checkpoint_fp = os.path.join(FLAGS.model_dir,
        'model.{epoch:03d}-{val_loss:.4f}.hdf5')

    model.fit(train_dataset,
        epochs=FLAGS.n_epochs,
        steps_per_epoch=train_steps,
        validation_data=dev_dataset,
        validation_steps=dev_steps,
        callbacks=[
            tf.keras.callbacks.TensorBoard(FLAGS.tb_log_dir),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_fp)
        ])


def main(_):

    if FLAGS.mode == 'train':
        train()


if __name__ == '__main__':

    flags.mark_flag_as_required('mode')
    flags.mark_flag_as_required('data_dir')

    app.run(main)