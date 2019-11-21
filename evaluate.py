import tensorflow as tf

_has_loss_func = False
try:
    from warprnnt_tensorflow import rnnt_loss
    _has_loss_func = True
except ImportError:
    pass

try:
    from .utils.data.common import preprocess_dataset
except ImportError:
    from utils.data.common import preprocess_dataset

def do_eval(model, vocab, dataset, batch_size, 
            shuffle_buffer_size=None, distribution_strategy=None):

    _dataset = preprocess_dataset(dataset, vocab, batch_size, 
        shuffle_buffer_size=shuffle_buffer_size)

    if distribution_strategy is not None:
        _dataset = distribution_strategy.experimental_distribute_dataset(
            _dataset)

    eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='eval_accuracy')

    @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float32),
                                  tf.TensorSpec([None, None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([2, None, None], tf.float32)])
    def eval_step(fb, labels, fb_lengths, labels_lengths, enc_state):

        pred_inp = labels[:, :-1]
        pred_out = labels[:, 1:]

        predictions, _ = model([fb, pred_inp, enc_state],
            training=False)
        
        if _has_loss_func:
            loss = warprnnt_tensorflow.rnnt_loss(predictions,
                                                 pred_out,
                                                 fb_lengths,
                                                 labels_lengths)
        else:
            loss = 0

        eval_loss(loss)
        eval_accuracy(pred_out, predictions)

    enc_state = model.initial_state(batch_size)

    for (inp, tar, inp_length, tar_length) in _dataset:

        if distribution_strategy is not None:
            distribution_strategy.experimental_run_v2(
                eval_step, args=(inp, tar, inp_length, tar_length, enc_state))
        else:
            eval_step(inp, tar, inp_length, tar_length, enc_state)

    return eval_loss.result(), eval_accuracy.result()