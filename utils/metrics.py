import tensorflow as tf

from . import decoding


def error_rate(y_true, decoded):

    y_true_shape = tf.shape(y_true)
    decoded_shape = tf.shape(decoded)

    max_length = tf.maximum(y_true_shape[-1], decoded_shape[-1])

    if y_true.dtype == tf.string:
        truth = string_to_sparse(y_true)
    else:
        truth = tf.sparse.from_dense(y_true)

    if decoded.dtype == tf.string:
        hypothesis = string_to_sparse(decoded)
    else:
        hypothesis = tf.sparse.from_dense(decoded)

    err = tf.edit_distance(hypothesis, truth, normalize=False)
    err_norm = err / tf.cast(max_length, dtype=tf.float32)

    return err_norm


def string_to_sparse(str_tensor):

    orig_shape = tf.cast(tf.shape(str_tensor), dtype=tf.int64)
    str_tensor = tf.squeeze(str_tensor, axis=0)

    indices = tf.concat([tf.zeros((orig_shape[-1], 1), dtype=tf.int64),
                         tf.expand_dims(tf.range(0, orig_shape[-1]), axis=-1)],
        axis=1)

    return tf.SparseTensor(indices=indices, values=str_tensor,
        dense_shape=orig_shape)


def token_error_rate(y_true, decoded, tok_fn, idx_to_text):

    text_true = idx_to_text(y_true)
    text_pred = idx_to_text(decoded)

    text_true.set_shape(())
    text_pred.set_shape(())

    tok_true = tok_fn(text_true)
    tok_pred = tok_fn(text_pred)

    tok_true = tf.expand_dims(tok_true, axis=0)
    tok_pred = tf.expand_dims(tok_pred, axis=0)

    return error_rate(tok_true, tok_pred)


def build_accuracy_fn(decode_fn):

    def Accuracy(inputs, y_true):

        # Decode functions only returns first result
        y_true = tf.expand_dims(y_true[0], axis=0)

        max_length = tf.shape(y_true)[1]

        decoded = decode_fn(inputs,
            max_length=max_length)

        return 1 - error_rate(y_true, decoded)

    return Accuracy


def build_wer_fn(decode_fn, idx_to_text):

    def WER(inputs, y_true):

        # Decode functions only returns first result
        y_true = y_true[0]

        max_length = tf.shape(y_true)[0]

        decoded = decode_fn(inputs,
            max_length=max_length)[0]

        return token_error_rate(y_true, decoded,
            tok_fn=lambda t: tf.strings.split(t, sep=' '),
            idx_to_text=idx_to_text)

    return WER
