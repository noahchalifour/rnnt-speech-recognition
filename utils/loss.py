from absl import logging
import tensorflow as tf

_has_loss_func = False
try:
    from warprnnt_tensorflow import rnnt_loss
    _has_loss_func = True
except ImportError:
    pass


def get_loss_fn(reduction_factor):

    def _fallback_loss(y_true, 
                       y_pred, 
                       spec_lengths,
                       label_lengths):
        logging.info('RNN-T loss function not found.')
        return y_pred

    if not _has_loss_func:
        return _fallback_loss

    def _loss_fn(y_true, 
                 y_pred,
                 spec_lengths,
                 label_lengths):
        y_true = tf.cast(y_true, dtype=tf.int32)
        if not tf.test.is_built_with_cuda():
            y_pred = tf.nn.log_softmax(y_pred)
        spec_lengths = spec_lengths // reduction_factor
        loss = rnnt_loss(y_pred, y_true,
            spec_lengths, label_lengths - 1)
        return loss

    return _loss_fn