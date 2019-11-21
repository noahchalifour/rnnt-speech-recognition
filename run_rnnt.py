from absl import app, logging, flags
import os
import json
import tensorflow as tf

from model import Transducer, load_model
from train import do_train
from evaluate import do_eval
from transcribe import transcribe_file
from utils import vocabulary
from utils.data import common as cmn_data_utils
import utils.data

def main(_):

    logging.info('Running with parameters:')
    logging.info(json.dumps(FLAGS.flag_values_dict(), indent=4))

    if os.path.exists(os.path.join(FLAGS.model_dir, 'config.json')):

        expect_partial = False
        if FLAGS.mode in ['transcribe-file']:
            expect_partial = True

        model, vocab = load_model(FLAGS.model_dir,
            checkpoint=FLAGS.checkpoint, expect_partial=expect_partial)

    else:

        if FLAGS.mode == 'eval' or FLAGS.mode == 'interactive':
            raise Exception('Model not found at path: {}'.format(
                FLAGS.model_dir))

        logging.info('Initializing model from scratch.')

        os.makedirs(FLAGS.model_dir, exist_ok=True)
        model_config_filepath = os.path.join(FLAGS.model_dir, 'config.json')

        vocab = vocabulary.init_vocab()
        vocabulary.save_vocab(vocab, os.path.join(FLAGS.model_dir, 'vocab'))

        model = Transducer(vocab_size=len(vocab),
                           encoder_layers=FLAGS.encoder_layers,
                           encoder_size=FLAGS.encoder_size,
                           pred_net_layers=FLAGS.pred_net_layers,
                           pred_net_size=FLAGS.pred_net_size,
                           joint_net_size=FLAGS.joint_net_size,
                           softmax_size=FLAGS.softmax_size)

        with open(model_config_filepath, 'w') as model_config:
            model_config.write(model.to_json())

        logging.info('Initialized model from scratch.')

    distribution_strategy = None

    if FLAGS.tpu is not None:

        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=FLAGS.tpu)
        distribution_strategy = tf.distribute.experimental.TPUStrategy(
            tpu_cluster_resolver=tpu_cluster_resolver)

    # if FLAGS.mode == 'optimize':

    #     model._set_inputs([tf.TensorSpec([None, None, None], dtype=tf.float32),
    #                        tf.TensorSpec([None, None], dtype=tf.int32),
    #                        tf.TensorSpec([2, None, None], dtype=tf.float32)])
        
    #     optmized_dir = './model/optimized'

    #     converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #     converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        
    #     tflite_quant_model = converter.convert()
    #     print(tflite_quant_model)

    elif FLAGS.mode == 'transcribe-file':

        transcription = transcribe_file(model, vocab, FLAGS.input)

        print('Input file: {}'.format(FLAGS.input))
        print('Transcription: {}'.format(transcription))

    else:

        if FLAGS.dataset_name == 'common-voice':
            data_utils = utils.data.common_voice

        train_dataset, dev_dataset = data_utils.create_datasets(FLAGS.dataset_path,
            max_data=FLAGS.max_data)

        if dev_dataset is None:
            dev_dataset = train_dataset.take(FLAGS.eval_size)
            train_dataset = train_dataset.skip(FLAGS.eval_size)

        if FLAGS.mode == 'eval':

            logging.info('Begin evaluation...')

            loss, acc = do_eval(model, vocab, dev_dataset,
                                batch_size=FLAGS.batch_size,
                                shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                                distribution_strategy=distribution_strategy)

            logging.info('Evaluation complete: Loss {} Accuracy {}'.format(
                loss, acc))

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
                     shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                     distribution_strategy=distribution_strategy)



def define_flags():

    FLAGS = flags.FLAGS

    flags.DEFINE_enum('mode', None, ['train', 'eval', 'transcribe-file'], 'Mode to run in.')
    flags.DEFINE_enum('dataset_name', None, ['common-voice'], 'Dataset to use.')
    flags.DEFINE_string('dataset_path', None, 'Dataset path.')
    flags.DEFINE_integer('max_data', None, 'Max size of data.')
    flags.DEFINE_string('input', None, 'Input file.')

    flags.DEFINE_integer('batch_size', 64, 'Batch size.')
    flags.DEFINE_integer('eval_size', 1000, 'Eval size.')
    flags.DEFINE_float('learning_rate', 1e-4, 'Training learning rate.')
    flags.DEFINE_integer('epochs', 20, 'Number of training epochs.')
    flags.DEFINE_string('model_dir', './model', 'Model output directory.')
    flags.DEFINE_integer('steps_per_log', 100, 'Number of steps between each log written.')
    flags.DEFINE_integer('steps_per_checkpoint', 1000, 'Number of steps between each checkpoint.')
    flags.DEFINE_string('checkpoint', None, 'Path of checkpoint to load (default to latest in \'model_dir\')')
    flags.DEFINE_string('tb_log_dir', './logs', 'Tensorboard log directory.')
    flags.DEFINE_integer('keep_top', 5, 'Maximum checkpoints to keep.')
    flags.DEFINE_integer('shuffle_buffer_size', None, 'Shuffle buffer size.')

    flags.DEFINE_integer('encoder_layers', 8, 'Number of encoder layers.')
    flags.DEFINE_integer('encoder_size', 2048, 'Units per encoder layer.')
    flags.DEFINE_integer('pred_net_layers', 2, 'Number of prediction network layers.')
    flags.DEFINE_integer('pred_net_size', 2048, 'Units per prediction network layer.')
    flags.DEFINE_integer('joint_net_size', 640, 'Joint network units.')
    flags.DEFINE_integer('softmax_size', 4096, 'Units in softmax layer.')

    flags.DEFINE_string('tpu', None, 'GCP TPU to use.')

    flags.mark_flags_as_required(['mode'])

    return FLAGS


if __name__ == '__main__':

    FLAGS = define_flags()
    app.run(main)