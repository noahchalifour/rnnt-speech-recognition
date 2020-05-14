from absl import flags, logging, app
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from datetime import datetime
import json
import re
import os
import time
import shutil

import tensorflow as tf
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(0)
# tf.random.set_seed(1234)

from utils import preprocessing, vocabulary, encoding, \
    metrics, decoding
from utils.loss import get_loss_fn
from utils import model as model_utils
from model import build_keras_model
from hparams import *

FLAGS = flags.FLAGS

# Required flags
flags.DEFINE_enum(
    'mode', None,
    ['train', 'eval', 'test'],
    'Mode to run.')
flags.DEFINE_string(
    'data_dir', None,
    'Input data directory.')

# Optional flags
flags.DEFINE_string(
    'tb_log_dir', './logs',
    'Directory to save Tensorboard logs.')
flags.DEFINE_string(
    'output_dir', './model',
    'Directory to save model.')
flags.DEFINE_string(
    'checkpoint', None,
    'Checkpoint to restore from.')
flags.DEFINE_integer(
    'batch_size', 32,
    'Training batch size.')
flags.DEFINE_integer(
    'n_epochs', 1000,
    'Number of training epochs.')
flags.DEFINE_integer(
    'steps_per_log', 1,
    'Number of steps between each log.')
flags.DEFINE_integer(
    'steps_per_checkpoint', 1000,
    'Number of steps between eval and checkpoint.')
flags.DEFINE_integer(
    'eval_size', None,
    'Max number of samples to use for eval.')
flags.DEFINE_list(
    'gpus', None,
    'GPUs to run training on.')
flags.DEFINE_bool(
    'fp16_run', False,
    'Run using 16-bit precision instead of 32-bit.')

def get_dataset(data_dir,
                name,
                batch_size,
                n_epochs,
                strategy=None,
                max_size=None):

    dataset = preprocessing.load_dataset(data_dir, name)

    if max_size is not None:
        dataset = dataset.take(max_size)

    dataset = dataset.padded_batch(
        batch_size, padded_shapes=(
            [-1, -1], [-1], [], [],
            [-1]
        )
    )

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if strategy is not None:
        dataset = strategy.experimental_distribute_dataset(dataset)

    return dataset


def configure_environment(gpu_names,
                          fp16_run):

    if fp16_run:
        print('Using 16-bit float precision.')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpu_names is not None and len(gpu_names) > 0:
        gpus = [x for x in gpus if x.name[len('/physical_device:'):] in gpu_names]

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
            #         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            logging.warn(str(e))

    if len(gpus) > 1:
        print('Running multi gpu: {}'.format(', '.join(gpu_names)))
        strategy = tf.distribute.MirroredStrategy(
            devices=gpu_names)
    else:
        device = gpus[0].name[len('/physical_device:'):]
        print('Running single gpu: {}'.format(device))
        strategy = tf.distribute.OneDeviceStrategy(
            device=device)

    dtype = tf.float16 if fp16_run else tf.float32

    return strategy, dtype


def setup_hparams(log_dir,
                  checkpoint):

    if checkpoint is not None:

        checkpoint_dir = os.path.dirname(os.path.realpath(checkpoint))
        hparams = model_utils.load_hparams(checkpoint_dir)

        tb_hparams = {}
        tb_keys = [
            HP_TOKEN_TYPE,
            HP_MEL_BINS,
            HP_FRAME_LENGTH,
            HP_FRAME_STEP,
            HP_HERTZ_LOW,
            HP_HERTZ_HIGH,
            HP_DOWNSAMPLE_FACTOR,
            HP_EMBEDDING_SIZE,
            HP_ENCODER_LAYERS,
            HP_ENCODER_SIZE,
            HP_PROJECTION_SIZE,
            HP_TIME_REDUCT_FACTOR,
            HP_TIME_REDUCT_INDEX,
            HP_PRED_NET_LAYERS,
            HP_PRED_NET_SIZE,
            HP_JOINT_NET_SIZE,
            HP_DROPOUT,
            HP_LEARNING_RATE
        ]

        for k, v in hparams.items():
            for tb_key in tb_keys:
                if k == tb_key.name:
                    tb_hparams[tb_key] = v

    else:

        tb_hparams = {

            HP_TOKEN_TYPE: HP_TOKEN_TYPE.domain.values[1],

            # Preprocessing
            HP_MEL_BINS: HP_MEL_BINS.domain.values[0],
            HP_FRAME_LENGTH: HP_FRAME_LENGTH.domain.values[0],
            HP_FRAME_STEP: HP_FRAME_STEP.domain.values[0],
            HP_HERTZ_LOW: HP_HERTZ_LOW.domain.values[0],
            HP_HERTZ_HIGH: HP_HERTZ_HIGH.domain.values[0],
            HP_DOWNSAMPLE_FACTOR: HP_DOWNSAMPLE_FACTOR.domain.values[0],

            # Model
            HP_EMBEDDING_SIZE: HP_EMBEDDING_SIZE.domain.values[0],
            HP_ENCODER_LAYERS: HP_ENCODER_LAYERS.domain.values[0],
            HP_ENCODER_SIZE: HP_ENCODER_SIZE.domain.values[0],
            HP_PROJECTION_SIZE: HP_PROJECTION_SIZE.domain.values[0],
            HP_TIME_REDUCT_INDEX: HP_TIME_REDUCT_INDEX.domain.values[0],
            HP_TIME_REDUCT_FACTOR: HP_TIME_REDUCT_FACTOR.domain.values[0],
            HP_PRED_NET_LAYERS: HP_PRED_NET_LAYERS.domain.values[0],
            HP_PRED_NET_SIZE: HP_PRED_NET_SIZE.domain.values[0],
            HP_JOINT_NET_SIZE: HP_JOINT_NET_SIZE.domain.values[0],
            HP_DROPOUT: HP_DROPOUT.domain.values[0],

            HP_LEARNING_RATE: HP_LEARNING_RATE.domain.values[0]

        }

    with tf.summary.create_file_writer(os.path.join(log_dir, 'hparams_tuning')).as_default():
        hp.hparams_config(
            hparams=[
                HP_TOKEN_TYPE,
                HP_VOCAB_SIZE,
                HP_ENCODER_LAYERS,
                HP_ENCODER_SIZE,
                HP_PROJECTION_SIZE,
                HP_TIME_REDUCT_INDEX,
                HP_TIME_REDUCT_FACTOR,
                HP_PRED_NET_LAYERS,
                HP_PRED_NET_SIZE,
                HP_JOINT_NET_SIZE,
                HP_DROPOUT
            ],
            metrics=[
                hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                hp.Metric(METRIC_WER, display_name='WER'),
            ],
        )

    return {k.name: v for k, v in tb_hparams.items()}, tb_hparams


def run_metrics(inputs,
                y_true,
                metrics,
                strategy=None):

    return {
        metric_fn.__name__: metric_fn(inputs, y_true)
        for metric_fn in metrics}


def run_training(model,
                 optimizer,
                 loss_fn,
                 train_dataset,
                 batch_size,
                 n_epochs,
                 checkpoint_template,
                 hparams,
                 noise=0,
                 # noise=0.075,
                 strategy=None,
                 steps_per_log=None,
                 steps_per_checkpoint=None,
                 eval_dataset=None,
                 train_metrics=[],
                 eval_metrics=[],
                 fp16_run=False):

    feat_size = hparams[HP_MEL_BINS.name] * hparams[HP_DOWNSAMPLE_FACTOR.name]

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, feat_size], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)]])
    def train_step(dist_inputs):
        def step_fn(inputs):

            (mel_specs, pred_inp,
             spec_lengths, label_lengths, labels) = inputs
            if noise > 0:
                mel_specs += tf.random.normal([mel_specs.shape[-1]],
                    mean=0, stddev=noise)

            with tf.GradientTape() as tape:
                outputs = model([mel_specs, pred_inp],
                    training=True)

                rnnt_loss = loss_fn(labels, outputs,
                    spec_lengths, label_lengths)

                if fp16_run:
                    rnnt_loss = optimizer.get_scaled_loss(rnnt_loss)

                loss = tf.reduce_sum(rnnt_loss) * (1. / batch_size)

            if train_metrics is not None:
                metric_results = run_metrics(mel_specs, labels,
                    metrics=train_metrics, strategy=strategy)

            gradients = tape.gradient(loss, model.trainable_variables)
            if fp16_run:
                gradients = optimizer.get_unscaled_gradients(gradients)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return rnnt_loss, metric_results

        loss, metrics_results = strategy.run(step_fn, args=(dist_inputs,))
        loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, loss, axis=0)
        metrics_results = {name: strategy.reduce(
            tf.distribute.ReduceOp.MEAN, result, axis=0) for name, result in metrics_results.items()}

        return loss, metrics_results

    def checkpoint_model():

        eval_start_time = time.time()

        eval_loss, eval_metrics_results = run_evaluate(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            eval_dataset=eval_dataset,
            batch_size=batch_size,
            hparams=hparams,
            strategy=strategy,
            metrics=eval_metrics)

        validation_log_str = 'VALIDATION RESULTS: Time: {:.4f}, Loss: {:.4f}'.format(
            time.time() - eval_start_time, eval_loss)
        for metric_name, metric_result in eval_metrics_results.items():
            validation_log_str += ', {}: {:.4f}'.format(metric_name, metric_result)
        print(validation_log_str)

        tf.summary.scalar(METRIC_EVAL_LOSS, eval_loss, step=global_step)
        if 'Accuracy' in eval_metrics_results:
            tf.summary.scalar(METRIC_EVAL_ACCURACY, eval_metrics_results['Accuracy'], step=global_step)
        if 'WER' in eval_metrics_results:
            tf.summary.scalar(METRIC_EVAL_WER, eval_metrics_results['WER'], step=global_step)

        checkpoint_filepath = checkpoint_template.format(
            step=global_step, val_loss=eval_loss)
        print('Saving checkpoint {}'.format(checkpoint_filepath))
        model.save_weights(checkpoint_filepath)


    with strategy.scope():

        print('Starting training.')

        global_step = 0

        for epoch in range(n_epochs):

            loss_object = tf.keras.metrics.Mean()
            metric_objects = {fn.__name__: tf.keras.metrics.Mean() for fn in train_metrics}

            for batch, inputs in enumerate(train_dataset):

                if global_step % steps_per_checkpoint == 0:
                    if eval_dataset is not None:
                        checkpoint_model()

                start_time = time.time()

                loss, metrics_results = train_step(inputs)

                step_time = time.time() - start_time

                loss_object(loss)
                for metric_name, metric_result in metrics_results.items():
                    metric_objects[metric_name](metric_result)

                if global_step % steps_per_log == 0:
                    log_str = 'Epoch: {}, Batch: {}, Global Step: {}, Step Time: {:.4f}, Loss: {:.4f}'.format(
                        epoch, batch, global_step, step_time, loss_object.result())
                    for metric_name, metric_object in metric_objects.items():
                        log_str += ', {}: {:.4f}'.format(metric_name, metric_object.result())
                    print(log_str)

                    tf.summary.scalar(METRIC_TRAIN_LOSS, loss_object.result(), step=global_step)
                    if 'Accuracy' in metric_objects:
                        tf.summary.scalar(METRIC_TRAIN_ACCURACY, metric_objects['Accuracy'].result(), step=global_step)

                global_step += 1

            epoch_end_log_str = 'EPOCH RESULTS: Loss: {:.4f}'.format(loss_object.result())
            for metric_name, metric_object in metric_objects.items():
                epoch_end_log_str += ', {}: {:.4f}'.format(metric_name, metric_object.result())
            print(epoch_end_log_str)

        checkpoint_model()


def run_evaluate(model,
                 optimizer,
                 loss_fn,
                 eval_dataset,
                 batch_size,
                 strategy,
                 hparams,
                 metrics=[],
                 fp16_run=False):

    feat_size = hparams[HP_MEL_BINS.name] * hparams[HP_DOWNSAMPLE_FACTOR.name]

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, feat_size], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)]])
    def eval_step(dist_inputs):
        def step_fn(inputs):
            (mel_specs, pred_inp,
             spec_lengths, label_lengths, labels) = inputs
            outputs = model([mel_specs, pred_inp],
                training=False)

            loss = loss_fn(labels, outputs,
                spec_lengths=spec_lengths,
                label_lengths=label_lengths)

            if fp16_run:
                loss = optimizer.get_scaled_loss(loss)

            if metrics is not None:
                metric_results = run_metrics(mel_specs, labels,
                    metrics=metrics, strategy=strategy)

            return loss, metric_results

        loss, metrics_results = strategy.run(step_fn, args=(dist_inputs,))
        loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, loss, axis=0)
        metrics_results = {name: strategy.reduce(
            tf.distribute.ReduceOp.MEAN, result, axis=0) for name, result in metrics_results.items()}

        return loss, metrics_results

    print('Performing evaluation.')

    loss_object = tf.keras.metrics.Mean()
    metric_objects = {fn.__name__: tf.keras.metrics.Mean() for fn in metrics}

    for batch, inputs in enumerate(eval_dataset):

        loss, metrics_results = eval_step(inputs)

        loss_object(loss)
        for metric_name, metric_result in metrics_results.items():
            metric_objects[metric_name](metric_result)

    metrics_final_results = {name: metric_object.result() for name, metric_object in metric_objects.items()}

    return loss_object.result(), metrics_final_results


def main(_):

    strategy, dtype = configure_environment(
        gpu_names=FLAGS.gpus,
        fp16_run=FLAGS.fp16_run)

    hparams, tb_hparams = setup_hparams(
        log_dir=FLAGS.tb_log_dir,
        checkpoint=FLAGS.checkpoint)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    if FLAGS.checkpoint is None:
        encoder_dir = FLAGS.data_dir
    else:
        encoder_dir = os.path.dirname(os.path.realpath(FLAGS.checkpoint))

    shutil.copy(
        os.path.join(encoder_dir, 'encoder.subwords'),
        os.path.join(FLAGS.output_dir, 'encoder.subwords'))

    encoder_fn, idx_to_text, vocab_size = encoding.get_encoder(
        encoder_dir=FLAGS.output_dir,
        hparams=hparams)

    if HP_VOCAB_SIZE.name not in hparams:
        hparams[HP_VOCAB_SIZE.name] = vocab_size

    with strategy.scope():

        model = build_keras_model(hparams,
            dtype=dtype)

        if FLAGS.checkpoint is not None:
            model.load_weights(FLAGS.checkpoint)
            logging.info('Restored weights from {}.'.format(FLAGS.checkpoint))

        model_utils.save_hparams(hparams, FLAGS.output_dir)

        optimizer = tf.keras.optimizers.SGD(hparams[HP_LEARNING_RATE.name],
            momentum=0.9)

        if FLAGS.fp16_run:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer,
                loss_scale='dynamic')

    logging.info('Using {} encoder with vocab size: {}'.format(
        hparams[HP_TOKEN_TYPE.name], vocab_size))

    loss_fn = get_loss_fn(
        reduction_factor=hparams[HP_TIME_REDUCT_FACTOR.name])

    decode_fn = decoding.greedy_decode_fn(model, hparams)

    accuracy_fn = metrics.build_accuracy_fn(decode_fn)
    wer_fn = metrics.build_wer_fn(decode_fn, idx_to_text)

    encoder = model.layers[2]
    prediction_network = model.layers[3]

    encoder.summary()
    prediction_network.summary()

    model.summary()

    dev_dataset = None
    if FLAGS.eval_size != 0:
        dev_dataset = get_dataset(FLAGS.data_dir, 'dev',
            batch_size=FLAGS.batch_size, n_epochs=FLAGS.n_epochs,
            strategy=strategy, max_size=FLAGS.eval_size)

    log_dir = os.path.join(FLAGS.tb_log_dir,
        datetime.now().strftime('%Y%m%d-%H%M%S'))

    with tf.summary.create_file_writer(log_dir).as_default():

        hp.hparams(tb_hparams)

        if FLAGS.mode == 'train':

            train_dataset = get_dataset(FLAGS.data_dir, 'train',
                batch_size=FLAGS.batch_size, n_epochs=FLAGS.n_epochs,
                strategy=strategy)

            os.makedirs(FLAGS.output_dir, exist_ok=True)
            checkpoint_template = os.path.join(FLAGS.output_dir,
                'checkpoint_{step}_{val_loss:.4f}.hdf5')

            run_training(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_dataset=train_dataset,
                batch_size=FLAGS.batch_size,
                n_epochs=FLAGS.n_epochs,
                checkpoint_template=checkpoint_template,
                hparams=hparams,
                strategy=strategy,
                steps_per_log=FLAGS.steps_per_log,
                steps_per_checkpoint=FLAGS.steps_per_checkpoint,
                eval_dataset=dev_dataset,
                train_metrics=[],
                eval_metrics=[accuracy_fn, wer_fn])

        elif FLAGS.mode == 'eval' or FLAGS.mode == 'test':

            if FLAGS.checkpoint is None:
                raise Exception('You must provide a checkpoint to perform eval.')

            if FLAGS.mode == 'test':
                dataset = get_dataset(FLAGS.data_dir, 'test',
                    batch_size=FLAGS.batch_size, n_epochs=FLAGS.n_epochs)
            else:
                dataset = dev_dataset

            eval_start_time = time.time()

            eval_loss, eval_metrics_results = run_evaluate(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                eval_dataset=dataset,
                batch_size=FLAGS.batch_size,
                hparams=hparams,
                strategy=strategy,
                metrics=[accuracy_fn, wer_fn],
                gpus=gpus)

            validation_log_str = 'VALIDATION RESULTS: Time: {:.4f}, Loss: {:.4f}'.format(
                time.time() - eval_start_time, eval_loss)
            for metric_name, metric_result in eval_metrics_results.items():
                validation_log_str += ', {}: {:.4f}'.format(metric_name, metric_result)

            print(validation_log_str)


if __name__ == '__main__':

    # tf.config.experimental_run_functions_eagerly(True)

    flags.mark_flag_as_required('mode')
    flags.mark_flag_as_required('data_dir')

    app.run(main)
