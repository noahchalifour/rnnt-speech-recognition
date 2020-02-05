import tensorflow as tf


def cer(y_true, y_pred):

    y_true = tf.cast(y_true, dtype=tf.int32)
    y_true = tf.squeeze(y_true)

    y_pred = y_pred[:, -1, :]

    return 1 - \
        tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)