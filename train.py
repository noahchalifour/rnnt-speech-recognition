from absl import logging
import time
import datetime
import os
import sys
import glob
import re
import tensorflow as tf

_has_loss_func = False
try:
    from warprnnt_tensorflow import rnnt_loss
    _has_loss_func = True
except ImportError:
    pass

try:
    from .utils.data.common import preprocess_dataset
    from .evaluate import do_eval
except ImportError:
    from utils.data.common import preprocess_dataset
    from evaluate import do_eval

def do_train(model, train_dataset, 
             optimizer, epochs, batch_size,
             eval_dataset=None, steps_per_checkpoint=None,
             checkpoint_path=None, steps_per_log=None,
             tb_log_dir=None, keep_top_n=None,
             shuffle_buffer_size=None,
             distribution_strategy=None, verbose=1):

    train_dataset = preprocess_dataset(train_dataset, model.vocab,
        batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    @tf.function(input_signature=[tf.TensorSpec([None, None, 80], tf.float32),
                                  tf.TensorSpec([None, None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([2, None, None], tf.float32)])
    def train_step(fb, labels, fb_lengths, labels_lengths, enc_state):

        pred_inp = labels[:, :-1]
        pred_out = labels[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = model([fb, pred_inp, enc_state],
                training=True)
            if len(tf.config.list_physical_devices('GPU')) == 0 and _has_loss_func:
                predictions = tf.nn.log_softmax(predictions)
            if _has_loss_func:
                loss = rnnt_loss(predictions, pred_out, fb_lengths, labels_lengths)
            else:
                loss = 0
                if verbose:
                    logging.info('Loss function not available, not computing gradients or optimizing.')

        if _has_loss_func:
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(pred_out, predictions[:, -1, :, :])

    global_step = model._checkpoint_step
    train_summary_writer, eval_summary_writer = None, None

    if tb_log_dir is not None:

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(tb_log_dir, current_time, 'train')
        eval_log_dir = os.path.join(tb_log_dir, current_time, 'eval')
        os.makedirs(train_log_dir)
        os.makedirs(eval_log_dir)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    enc_state = model.initial_state(batch_size)

    for epoch in range(epochs):

        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (fb, labels, fb_lengths, labels_lengths)) in enumerate(train_dataset):

            if distribution_strategy is not None:
                distribution_strategy.experimental_run_v2(
                    train_step, args=(fb, labels, fb_lengths, labels_lengths, enc_state))
            else:
                train_step(fb, labels, fb_lengths, labels_lengths, enc_state)

            if steps_per_log is not None and global_step % steps_per_log == 0:
                if verbose:
                    logging.info('Epoch {} Batch {} Global Step {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, global_step, train_loss.result(), train_accuracy.result()))
                if train_summary_writer is not None:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(), step=global_step)
                        tf.summary.scalar('accuracy', train_accuracy.result(), step=global_step)
            
            if checkpoint_path is not None and steps_per_checkpoint is not None and global_step != 0 and global_step % steps_per_checkpoint == 0:
                eval_loss = 'na'
                if eval_dataset is not None:
                    if verbose:
                        logging.info('Evaluating model...')
                    eval_loss, eval_acc = do_eval(model, 
                        eval_dataset, batch_size, shuffle_buffer_size,
                        distribution_strategy=distribution_strategy)
                    if verbose:
                        logging.info('Evaluation result: Loss: {} Accuracy {}'.format(
                            eval_loss, eval_acc))
                    if eval_summary_writer is not None:
                        with eval_summary_writer.as_default():
                            tf.summary.scalar('loss', eval_loss, step=global_step)
                            tf.summary.scalar('accuracy', eval_acc, step=global_step)
                    eval_loss = '{:.4f}'.format(eval_loss)
                _checkpoint_path = os.path.join(checkpoint_path, 'ckpt_{}_{}'.format(
                    global_step, eval_loss))
                model.save_weights(_checkpoint_path, save_format='tf')
                if verbose:
                    logging.info('Saving checkpoint for step {} at {}'.format(global_step,
                                                                    _checkpoint_path))

                # Keep only top n checkpoints (sorted by eval loss)
                if keep_top_n is not None:
                    checkpoints = list(set([re.findall(r'(ckpt_\d+_\d+\.\d+)', x)[0] for x in os.listdir(checkpoint_path) 
                                            if x != 'checkpoint' and '_na' not in x]))
                    _checkpoints = {}
                    for ckpt in checkpoints:
                        try:
                            _checkpoints[ckpt] = float(re.findall(r'_(\d+\.\d+)', ckpt)[0])
                        except Exception:
                            pass
                    sorted_ckpts = sorted(_checkpoints.items(), key=lambda kv: kv[1])
                    for ckpt in sorted_ckpts[keep_top_n:]:
                        if verbose:
                            logging.info('Deleting checkpoint {}'.format(ckpt[0]))
                        _ckpt = os.path.join(checkpoint_path, ckpt[0])
                        ckpt_files = glob.glob(_ckpt + '*')
                        for f in ckpt_files:
                            os.remove(f)

                    # Update checkpoint file
                    if len(sorted_ckpts) > 0:
                        with open(os.path.join(checkpoint_path, 'checkpoint'), 'w') as f:
                            f.write('model_checkpoint_path: "{}"\nall_model_checkpoint_paths: "{}"'.format(
                                sorted_ckpts[0][0], sorted_ckpts[0][0]))
                        if verbose:
                            logging.info('Updated best checkpoint to {}'.format(sorted_ckpts[0][0]))

            global_step += 1
            tf.keras.backend.clear_session()

        if verbose:
            logging.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                train_loss.result(), train_accuracy.result()))
            logging.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    