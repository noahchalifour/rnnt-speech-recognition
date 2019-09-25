from absl import app, logging, flags
import os
import tensorflow as tf

from model import transducer
from train import do_train
from utils import vocabulary
from utils.data import common as cmn_data_utils
import utils.data

def main(_):

    if FLAGS.dataset_name == 'common-voice':
        data_utils = utils.data.common_voice

    vocab = vocabulary.init_vocab()

    model = transducer(vocab_size=len(vocab),
                       encoder_layers=FLAGS.encoder_layers,
                       encoder_size=FLAGS.encoder_size,
                       pred_net_layers=FLAGS.pred_net_layers,
                       pred_net_size=FLAGS.pred_net_size,
                       joint_net_size=FLAGS.joint_net_size,
                       softmax_size=FLAGS.softmax_size)

    if FLAGS.mode == 'realtime':

        pass

    else:

        train_dataset, dev_dataset = data_utils.create_datasets(FLAGS.dataset_path,
            max_data=FLAGS.max_data)

        if dev_dataset is None:
            dev_dataset = train_dataset.take(FLAGS.eval_size)
            train_dataset = train_dataset.skip(FLAGS.eval_size)

        if FLAGS.mode == 'eval':

            pass

        else:

            optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

            checkpoints_path = os.path.join(FLAGS.model_dir, 'checkpoints')
            os.makedirs(checkpoints_path, exist_ok=True)

            do_train(model, vocab, train_dataset, optimizer,
                     FLAGS.epochs, FLAGS.batch_size,
                     eval_dataset=dev_dataset,
                     steps_per_checkpoint=FLAGS.steps_per_checkpoint,
                     checkpoint_path=checkpoints_path,
                     steps_per_log=FLAGS.steps_per_log,
                     tb_log_dir=FLAGS.tb_log_dir,
                     keep_top_n=FLAGS.keep_top,
                     shuffle_buffer_size=FLAGS.shuffle_buffer_size)



def define_flags():

    FLAGS = flags.FLAGS

    flags.DEFINE_enum('mode', None, ['train', 'eval', 'realtime'], 'Mode to run in.')
    flags.DEFINE_enum('dataset_name', None, 'common-voice', 'Dataset to use.')
    flags.DEFINE_string('dataset_path', None, 'Dataset path.')
    flags.DEFINE_integer('max_data', None, 'Max size of data.')

    flags.DEFINE_integer('batch_size', 64, 'Batch size.')
    flags.DEFINE_integer('eval_size', 1000, 'Eval size.')
    flags.DEFINE_float('learning_rate', 1e-4, 'Training learning rate.')
    flags.DEFINE_integer('epochs', 20, 'Number of training epochs.')
    flags.DEFINE_string('model_dir', './model', 'Model output directory.')
    flags.DEFINE_integer('steps_per_log', 1, 'Number of steps between each log written.')
    flags.DEFINE_integer('steps_per_checkpoint', 10, 'Number of steps between each checkpoint.')
    flags.DEFINE_string('checkpoint', None, 'Path of checkpoint to load (default to latest in \'model_dir\')')
    flags.DEFINE_string('tb_log_dir', './logs', 'Tensorboard log directory.')
    flags.DEFINE_integer('keep_top', 5, 'Maximum checkpoints to keep.')
    flags.DEFINE_integer('shuffle_buffer_size', None, 'Shuffle buffer size.')

    flags.DEFINE_integer('encoder_layers', 1, 'Number of encoder layers.')
    flags.DEFINE_integer('encoder_size', 100, 'Units per encoder layer.')
    flags.DEFINE_integer('pred_net_layers', 1, 'Number of prediction network layers.')
    flags.DEFINE_integer('pred_net_size', 100, 'Units per prediction network layer.')
    flags.DEFINE_integer('joint_net_size', 100, 'Joint network units.')
    flags.DEFINE_integer('softmax_size', 30, 'Units in softmax layer.')

    flags.mark_flags_as_required(['mode'])

    return FLAGS


if __name__ == '__main__':

    FLAGS = define_flags()
    app.run(main)