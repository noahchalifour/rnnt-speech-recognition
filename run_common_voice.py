from absl import flags, logging, app
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from datetime import datetime
import json
import os
import re
import time
import shutil
import tensorflow as tf

from utils.data import common_voice
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
    'gpus', [],
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

    with open(os.path.join(data_dir, '{}-specs.json'.format(name)), 'r') as f:
        dataset_specs = json.load(f)

    return dataset, dataset_specs


def configure_environment(fp16_run):

    if fp16_run:
        print('Using 16-bit float precision.')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logging.warn(str(e))


def run_metrics(inputs, y_true, metrics):

    results = {}

    for metric_fn in metrics:
        results[metric_fn.__name__] = metric_fn(inputs, y_true)
        # results[metric_fn.__name__] = tf.constant([0], dtype=tf.float32)

    return results


def run_training(model, 
                 optimizer, 
                 loss_fn, 
                 train_dataset,
                 batch_size,
                 n_epochs,
                 checkpoint_template,
                 strategy=None,
                 steps_per_log=None,
                 steps_per_checkpoint=None,
                 eval_dataset=None,
                 train_metrics=[],
                 eval_metrics=[],
                 fp16_run=False,
                 gpus=[]):

    @tf.function(experimental_relax_shapes=True)
    def train_step(dist_inputs):
        def step_fn(inputs):
            (mel_specs, pred_inp, 
             spec_lengths, label_lengths, labels) = inputs
            with tf.GradientTape() as tape:
                outputs = model([mel_specs, pred_inp], 
                    training=True)

                loss = loss_fn(labels, outputs,
                    spec_lengths, label_lengths)
                loss *= (1. / batch_size)

                if fp16_run:
                    loss = optimizer.get_scaled_loss(loss)

            if train_metrics is not None:
                metric_results = run_metrics(mel_specs, labels,
                    metrics=train_metrics)
                metric_results = {name: result * (1. / max(len(gpus), 1)) for name, result in metric_results.items()}

            gradients = tape.gradient(loss, model.trainable_variables)
            if fp16_run:
                gradients = optimizer.get_unscaled_gradients(gradients)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return loss, metric_results

        if strategy is not None:
            losses, metrics_results = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
            mean_loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, losses, axis=0)
            mean_metrics = {name: strategy.reduce(
                tf.distribute.ReduceOp.SUM, result, axis=0) for name, result in metrics_results.items()}
        else:
            losses, metrics_results = step_fn(dist_inputs)
            mean_loss = tf.reduce_sum(losses, axis=0)
            mean_metrics = {name: tf.reduce_sum(result, axis=0) for name, result in metrics_results.items()}
        
        return mean_loss, mean_metrics

    def train():

        print('Starting training.')

        global_step = 0

        for epoch in range(n_epochs):
            
            loss_object = tf.keras.metrics.Mean()
            metric_objects = {fn.__name__: tf.keras.metrics.Mean() for fn in train_metrics}

            for batch, inputs in enumerate(train_dataset):

                if global_step % steps_per_checkpoint == 0:
                    if eval_dataset is not None:
                        eval_start_time = time.time()

                        eval_loss, eval_metrics_results = run_evaluate(
                            model=model,
                            optimizer=optimizer, 
                            loss_fn=loss_fn,
                            eval_dataset=eval_dataset,
                            batch_size=batch_size,
                            strategy=strategy,
                            metrics=eval_metrics,
                            gpus=gpus)

                        validation_log_str = 'VALIDATION RESULTS: Time: {:.4f}, Loss: {:.4f}'.format(
                            time.time() - eval_start_time, eval_loss)
                        for metric_name, metric_result in eval_metrics_results.items():
                            validation_log_str += ', {}: {:.4f}'.format(metric_name, metric_result)
                        print(validation_log_str)

                        tf.summary.scalar(METRIC_EVAL_LOSS, eval_loss, step=global_step)
                        if 'Accuracy' in eval_metrics_results:
                            tf.summary.scalar(METRIC_EVAL_ACCURACY, eval_metrics_results['Accuracy'], step=global_step)
                        if 'CER' in eval_metrics_results:
                            tf.summary.scalar(METRIC_EVAL_CER, eval_metrics_results['CER'], step=global_step)
                        if 'WER' in eval_metrics_results:
                            tf.summary.scalar(METRIC_EVAL_WER, eval_metrics_results['WER'], step=global_step)

                    checkpoint_filepath = checkpoint_template.format(
                        step=global_step, val_loss=eval_loss)
                    print('Saving checkpoint {}'.format(checkpoint_filepath))
                    model.save_weights(checkpoint_filepath)

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

    if strategy is not None:
        with strategy.scope():
            train()
    else:
        train()


def run_evaluate(model,
                 optimizer,
                 loss_fn, 
                 eval_dataset,
                 batch_size,
                 strategy,
                 metrics=[],
                 fp16_run=False,
                 gpus=[]):

    @tf.function(experimental_relax_shapes=True)
    def eval_step(dist_inputs):
        def step_fn(inputs):
            (mel_specs, pred_inp, 
             spec_lengths, label_lengths, labels) = inputs

            outputs = model([mel_specs, pred_inp], 
                training=False)

            loss = loss_fn(labels, outputs,
                spec_lengths=spec_lengths,
                label_lengths=label_lengths)
            loss *= (1. / batch_size)

            if fp16_run:
                loss = optimizer.get_scaled_loss(loss)

            if metrics is not None:
                metric_results = run_metrics(mel_specs, labels,
                    metrics=metrics)
                metric_results = {name: result * (1. / max(len(gpus), 1)) for name, result in metric_results.items()}

            return loss, metric_results

        losses, metrics_results = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
        mean_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, losses, axis=0)
        mean_metrics = {name: strategy.reduce(
            tf.distribute.ReduceOp.SUM, result, axis=0) for name, result in metrics_results.items()}
        
        return mean_loss, mean_metrics

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

    configure_environment(FLAGS.fp16_run)

    hparams = {

        HP_TOKEN_TYPE: HP_TOKEN_TYPE.domain.values[1],

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
        HP_PROJECTION_SIZE: HP_PROJECTION_SIZE.domain.values[0],
        HP_TIME_REDUCT_INDEX: HP_TIME_REDUCT_INDEX.domain.values[0],
        HP_TIME_REDUCT_FACTOR: HP_TIME_REDUCT_FACTOR.domain.values[0],
        HP_PRED_NET_LAYERS: HP_PRED_NET_LAYERS.domain.values[0],
        HP_PRED_NET_SIZE: HP_PRED_NET_SIZE.domain.values[0],
        HP_JOINT_NET_SIZE: HP_JOINT_NET_SIZE.domain.values[0],

        HP_LEARNING_RATE: HP_LEARNING_RATE.domain.values[0]

    }

    with tf.summary.create_file_writer(os.path.join(FLAGS.tb_log_dir, 'hparams_tuning')).as_default():
        hp.hparams_config(
            hparams=[
                HP_TOKEN_TYPE,
                HP_VOCAB_SIZE,
                HP_EMBEDDING_SIZE,
                HP_ENCODER_LAYERS,
                HP_ENCODER_SIZE,
                HP_PROJECTION_SIZE,
                HP_TIME_REDUCT_INDEX,
                HP_TIME_REDUCT_FACTOR,
                HP_PRED_NET_LAYERS,
                HP_PRED_NET_SIZE,
                HP_JOINT_NET_SIZE
            ],
            metrics=[
                hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                hp.Metric(METRIC_CER, display_name='CER'),
                hp.Metric(METRIC_WER, display_name='WER'),
            ],
        )

    _hparams = {k.name: v for k, v in hparams.items()}

    if len(FLAGS) == 0:
        gpus = [x.name.strip('/physical_device:')
                for x in tf.config.experimental.list_physical_devices('GPU')]
    else:
        gpus = ['GPU:' + str(gpu_id) for gpu_id in FLAGS.gpus]

    strategy = tf.distribute.MirroredStrategy(
        devices=gpus)
    # strategy = None

    dtype = tf.float32
    if FLAGS.fp16_run:
        dtype = tf.float16

    # initializer = tf.keras.initializers.RandomUniform(
    #     minval=-0.1, maxval=0.1)
    initializer = None

    if FLAGS.checkpoint is not None:
        
        checkpoint_dir = os.path.dirname(os.path.realpath(FLAGS.checkpoint))

        _hparams = model_utils.load_hparams(checkpoint_dir)
        encoder_fn, idx_to_text, vocab_size = encoding.load_encoder(checkpoint_dir,
            hparams=_hparams)

        if strategy is not None:
            with strategy.scope():
                model = build_keras_model(_hparams, 
                    initializer=initializer, dtype=dtype)
                model.load_weights(FLAGS.checkpoint)
        else:
            model = build_keras_model(_hparams, 
                initializer=initializer, dtype=dtype)
            model.load_weights(FLAGS.checkpoint)

        logging.info('Restored weights from {}.'.format(FLAGS.checkpoint))

    else:
        
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        
        shutil.copy(
            os.path.join(FLAGS.data_dir, 'encoder.subwords'),
            os.path.join(FLAGS.output_dir, 'encoder.subwords'))

        encoder_fn, idx_to_text, vocab_size = encoding.load_encoder(FLAGS.output_dir,
            hparams=_hparams)
        _hparams[HP_VOCAB_SIZE.name] = vocab_size

        if strategy is not None:
            with strategy.scope():
                model = build_keras_model(_hparams, 
                    initializer=initializer, dtype=dtype)
        else:
            model = build_keras_model(_hparams, 
                initializer=initializer, dtype=dtype)
        model_utils.save_hparams(_hparams, FLAGS.output_dir)

    logging.info('Using {} encoder with vocab size: {}'.format(
        _hparams[HP_TOKEN_TYPE.name], vocab_size))

    loss_fn = get_loss_fn(
        reduction_factor=_hparams[HP_TIME_REDUCT_FACTOR.name])

    start_token = encoder_fn('')[0]
    decode_fn = decoding.greedy_decode_fn(model,
        start_token=start_token)

    accuracy_fn = metrics.build_accuracy_fn(decode_fn)
    cer_fn = metrics.build_cer_fn(decode_fn, idx_to_text)
    wer_fn = metrics.build_wer_fn(decode_fn, idx_to_text)

    optimizer = tf.keras.optimizers.SGD(_hparams[HP_LEARNING_RATE.name],
        momentum=0.9)

    if FLAGS.fp16_run:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, 
            loss_scale='dynamic')

    encoder = model.layers[2]
    prediction_network = model.layers[3]

    encoder.summary()
    prediction_network.summary()

    model.summary()

    dev_dataset, _ = get_dataset(FLAGS.data_dir, 'dev',
        batch_size=FLAGS.batch_size, n_epochs=FLAGS.n_epochs,
        strategy=strategy, max_size=FLAGS.eval_size)
    # dev_steps = dev_specs['size'] // FLAGS.batch_size

    log_dir = os.path.join(FLAGS.tb_log_dir,
        datetime.now().strftime('%Y%m%d-%H%M%S'))

    with tf.summary.create_file_writer(log_dir).as_default():

        hp.hparams(hparams)

        if FLAGS.mode == 'train':

            train_dataset, _ = get_dataset(FLAGS.data_dir, 'train',
                batch_size=FLAGS.batch_size, n_epochs=FLAGS.n_epochs,
                strategy=strategy)
            # train_steps = train_specs['size'] // FLAGS.batch_size
        
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
                strategy=strategy,
                steps_per_log=FLAGS.steps_per_log,
                steps_per_checkpoint=FLAGS.steps_per_checkpoint,
                eval_dataset=dev_dataset,
                train_metrics=[],
                eval_metrics=[accuracy_fn, cer_fn, wer_fn],
                gpus=gpus)

        elif FLAGS.mode == 'eval' or FLAGS.mode == 'test':

            if FLAGS.checkpoint is None:
                raise Exception('You must provide a checkpoint to perform eval.')

            if FLAGS.mode == 'test':
                dataset, test_specs = get_dataset(FLAGS.data_dir, 'test',
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
                strategy=strategy,
                metrics=[accuracy_fn, cer_fn, wer_fn],
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