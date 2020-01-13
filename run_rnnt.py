from absl import app, logging, flags
import os
import json
import pyaudio
import numpy as np
import tensorflow as tf

from model import Transducer, load_model
from train import do_train
from evaluate import do_eval
from transcribe import transcribe_file, transcribe_stream
from utils import vocabulary
from utils.data import common as cmn_data_utils
import utils.data


def load_datasets():

    if FLAGS.dataset_name == 'common-voice':
        data_utils = utils.data.common_voice

    return data_utils.create_datasets(FLAGS.dataset_path,
        max_data=FLAGS.max_data)


def main(_):

    logging.info('Running with parameters:')
    logging.info(json.dumps(FLAGS.flag_values_dict(), indent=4))

    if os.path.exists(os.path.join(FLAGS.model_dir, 'config.json')):

        expect_partial = False
        if FLAGS.mode in ['transcribe-file', 'realtime', 'export']:
            expect_partial = True

        model = load_model(FLAGS.model_dir,
            checkpoint=FLAGS.checkpoint, expect_partial=expect_partial)

    else:

        if FLAGS.mode in ['eval', 'transcribe-file', 'realtime']:
            raise Exception('Model not found at path: {}'.format(
                FLAGS.model_dir))

        logging.info('Initializing model from scratch.')

        os.makedirs(FLAGS.model_dir, exist_ok=True)
        model_config_filepath = os.path.join(FLAGS.model_dir, 'config.json')

        vocab = vocabulary.init_vocab()
        vocabulary.save_vocab(vocab, os.path.join(FLAGS.model_dir, 'vocab'))

        model = Transducer(vocab=vocab,
                           encoder_layers=FLAGS.encoder_layers,
                           encoder_size=FLAGS.encoder_size,
                           pred_net_layers=FLAGS.pred_net_layers,
                           joint_net_size=FLAGS.joint_net_size,
                           softmax_size=FLAGS.softmax_size)

        model.save_config(model_config_filepath)

        logging.info('Initialized model from scratch.')

    distribution_strategy = None

    if FLAGS.tpu is not None:

        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=FLAGS.tpu)
        distribution_strategy = tf.distribute.experimental.TPUStrategy(
            tpu_cluster_resolver=tpu_cluster_resolver)

    if FLAGS.mode == 'export':
        
        # saved_model_dir = os.path.join(FLAGS.model_dir, 'saved_model')
        # os.makedirs(saved_model_dir, exist_ok=True)
        
        # all_versions = [int(ver) for ver in os.listdir(saved_model_dir)]

        # if len(all_versions) > 0:
        #     version = max(all_versions) + 1
        # else:
        #     version = 1

        # export_path = os.path.join(saved_model_dir, str(version))
        # os.makedirs(export_path)

        # tf.saved_model.save(model, export_path, signatures={
        #     'serving_default': model.predict
        # })

        # print(model.predict(tf.zeros((1, 1024)), tf.constant([16000]), tf.constant(['hell']), tf.zeros((1, 2, 1, 2048))))

        tflite_dir = os.path.join(FLAGS.model_dir, 'lite')
        os.makedirs(tflite_dir, exist_ok=True)

        concrete_func = model.predict.get_concrete_function(
            audio=tf.TensorSpec([1, 1024], dtype=tf.float32),
            sr=tf.TensorSpec([1], dtype=tf.int32),
            pred_inp=tf.TensorSpec([1], dtype=tf.string),
            enc_state=tf.TensorSpec([1, 2, 1, model.encoder_size], dtype=tf.float32))

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        converter.allow_custom_ops = True

        # def representative_dataset_gen():
        #     dataset, _ = load_datasets()
        #     for i in range(10):
        #         yield [next(dataset)]

        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8

        tflite_quant_model = converter.convert()
        
        with open(os.path.join(tflite_dir, 'model.tflite'), 'wb') as f:
            f.write(tflite_quant_model)

        print('Exported model to TFLite.')

    elif FLAGS.mode == 'transcribe-file':

        transcription = transcribe_file(model, FLAGS.input)

        print('Input file: {}'.format(FLAGS.input))
        print('Transcription: {}'.format(transcription))

    elif FLAGS.mode == 'realtime':

        audio_buf = []
        last_result = None

        def stream_callback(in_data, frame_count, time_info, status):
            audio_buf.append(in_data)
            return None, pyaudio.paContinue

        def audio_gen():
            while True:
                if len(audio_buf) > 0:
                    audio_data = audio_buf[0]
                    audio_arr = np.frombuffer(audio_data, dtype=np.float32)
                    yield audio_arr

        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 16000
        CHUNK = 2048

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK,
                            stream_callback=stream_callback)
        
        stream.start_stream()

        outputs = transcribe_stream(model, audio_gen(), RATE)

        print('Transcribing live audio (press CTRL+C to stop)...')

        for (output, is_final) in outputs:
            if output != last_result and output != '' and not is_final:
                print('Partial Result: {}'.format(output))
                last_result = output
            if is_final:
                print('# Final Result: {}'.format(output))
                last_result = None

    else:

        train_dataset, dev_dataset = load_datasets()

        if dev_dataset is None:
            dev_dataset = train_dataset.take(FLAGS.eval_size)
            train_dataset = train_dataset.skip(FLAGS.eval_size)

        if FLAGS.eval_size:
            dev_dataset = dev_dataset.take(FLAGS.eval_size)

        if FLAGS.mode == 'eval':

            logging.info('Begin evaluation...')

            loss, acc = do_eval(model, dev_dataset,
                                batch_size=FLAGS.batch_size,
                                shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                                distribution_strategy=distribution_strategy)

            logging.info('Evaluation complete: Loss {} Accuracy {}'.format(
                loss, acc))

        else:

            optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

            checkpoints_path = os.path.join(FLAGS.model_dir, 'checkpoints')
            os.makedirs(checkpoints_path, exist_ok=True)

            do_train(model, train_dataset, optimizer,
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

    flags.DEFINE_enum('mode', None, ['train', 'eval', 'transcribe-file', 'realtime', 'export'], 'Mode to run in.')
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
    flags.DEFINE_integer('joint_net_size', 640, 'Joint network units.')
    flags.DEFINE_integer('softmax_size', 4096, 'Units in softmax layer.')

    flags.DEFINE_string('tpu', None, 'GCP TPU to use.')

    flags.mark_flags_as_required(['mode'])

    return FLAGS


if __name__ == '__main__':

    FLAGS = define_flags()
    app.run(main)