"""Metrics module for speech recognition evaluation.

This module provides functions for calculating various metrics commonly used in speech recognition tasks,
including character error rate (CER), word error rate (WER), and token error rate. It also provides
utility functions for building custom metric functions that can be used with TensorFlow models.
"""

import tensorflow as tf


def error_rate(y_true, decoded):
    """Calculate the normalized edit distance (Levenshtein distance) between true and predicted sequences.

    This function computes the edit distance between the true sequence and the decoded (predicted)
    sequence, and normalizes it by the maximum length of both sequences.

    Args:
        y_true: Tensor containing the true/reference sequences. Can be dense tensor or string tensor.
        decoded: Tensor containing the decoded/predicted sequences. Can be dense tensor or string tensor.

    Returns:
        Tensor of normalized edit distances (between 0 and 1).
    """
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
    """Convert a string tensor to a sparse tensor format.

    This function is used to convert string tensors to the sparse tensor format
    required by TensorFlow's edit_distance function.

    Args:
        str_tensor: A tensor of strings to convert to sparse format.

    Returns:
        A SparseTensor representation of the input string tensor.
    """
    orig_shape = tf.cast(tf.shape(str_tensor), dtype=tf.int64)
    str_tensor = tf.squeeze(str_tensor, axis=0)

    indices = tf.concat(
        [
            tf.zeros((orig_shape[-1], 1), dtype=tf.int64),
            tf.expand_dims(tf.range(0, orig_shape[-1]), axis=-1),
        ],
        axis=1,
    )

    return tf.SparseTensor(indices=indices, values=str_tensor, dense_shape=orig_shape)


def token_error_rate(y_true, decoded, tok_fn, idx_to_text):
    """Calculate the token error rate between true and predicted sequences.

    This function first converts indices to text using the provided idx_to_text function,
    then tokenizes the text using the provided tok_fn function, and finally computes
    the error rate between the tokenized sequences.

    Args:
        y_true: Tensor containing the true/reference sequence indices.
        decoded: Tensor containing the decoded/predicted sequence indices.
        tok_fn: Function to tokenize text strings (e.g., split by spaces for word-level tokens).
        idx_to_text: Function to convert indices to text strings.

    Returns:
        Tensor of token error rate (between 0 and 1).
    """
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
    """Build an accuracy function using the provided decode function.

    This function creates and returns an accuracy function that uses the provided
    decode function to convert model outputs to predictions, and then calculates
    accuracy as 1 minus the character error rate.

    Args:
        decode_fn: A function that takes model inputs and returns decoded predictions.

    Returns:
        A function that calculates accuracy between model inputs and true labels.
    """

    def Accuracy(inputs, y_true):
        """Calculate accuracy between model inputs and true labels.

        Args:
            inputs: The model inputs to decode.
            y_true: The true labels to compare against.

        Returns:
            Accuracy score (between 0 and 1).
        """
        # Decode functions only returns first result
        y_true = tf.expand_dims(y_true[0], axis=0)

        max_length = tf.shape(y_true)[1]

        decoded = decode_fn(inputs, max_length=max_length)

        return 1 - error_rate(y_true, decoded)

    return Accuracy


def build_wer_fn(decode_fn, idx_to_text):
    """Build a Word Error Rate (WER) function using the provided decode and text conversion functions.

    This function creates and returns a WER function that uses the provided decode function
    to convert model outputs to predictions, and the idx_to_text function to convert indices
    to text for word-level error rate calculation.

    Args:
        decode_fn: A function that takes model inputs and returns decoded predictions.
        idx_to_text: A function that converts indices to text strings.

    Returns:
        A function that calculates WER between model inputs and true labels.
    """

    def WER(inputs, y_true):
        """Calculate Word Error Rate between model inputs and true labels.

        Args:
            inputs: The model inputs to decode.
            y_true: The true labels to compare against.

        Returns:
            Word Error Rate score (between 0 and 1).
        """
        # Decode functions only returns first result
        y_true = y_true[0]

        max_length = tf.shape(y_true)[0]

        decoded = decode_fn(inputs, max_length=max_length)[0]

        return token_error_rate(
            y_true,
            decoded,
            tok_fn=lambda t: tf.strings.split(t, sep=" "),
            idx_to_text=idx_to_text,
        )

    return WER
