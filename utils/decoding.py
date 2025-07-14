"""
Decoding utilities for RNN-Transducer (RNN-T) speech recognition models.

This module provides functions for decoding outputs from an RNN-T model,
including joint network operations and greedy decoding implementation.
The decoding process converts the model's outputs into a sequence of tokens
that represent the transcription of the input audio.
"""

import tensorflow as tf

from hparams import HP_DOWNSAMPLE_FACTOR, HP_MEL_BINS


def joint(model, f, g):
    """
    Combines encoder and decoder outputs using the joint network.

    This function implements the joint network operation in RNN-T, which combines
    the encoder output (f) and the prediction network output (g) to produce logits
    for the next token prediction.

    Args:
        model: The RNN-T model containing the joint network layers
        f: Encoder outputs, shape [batch_size, time_steps, hidden_dim]
        g: Prediction network outputs, shape [batch_size, seq_len, hidden_dim]

    Returns:
        Joint network outputs (logits), shape [batch_size, vocab_size]
    """
    dense_1 = model.layers[-2]
    dense_2 = model.layers[-1]

    joint_inp = (
        tf.expand_dims(f, axis=2)  # [B, T, V] => [B, T, 1, V]
        + tf.expand_dims(g[:, -1, :], axis=1)
    )  # [B, U, V] => [B, 1, U, V]

    outputs = dense_1(joint_inp)
    outputs = dense_2(outputs)

    return outputs[:, 0, 0, :]


def greedy_decode_fn(model, hparams):
    """
    Creates a function that performs greedy decoding for RNN-T model inference.

    This function returns a TensorFlow function that performs greedy decoding
    on the output of an RNN-T model. The decoding process iterates through
    encoder outputs and predicts tokens until an end token is reached or
    maximum length is exceeded.

    Args:
        model: The RNN-T model to use for decoding
        hparams: Hyperparameters dictionary containing model configuration

    Returns:
        A TensorFlow function that takes model inputs and performs greedy decoding
    """
    # NOTE: Only the first input is decoded

    encoder = model.layers[2]
    prediction_network = model.layers[3]

    start_token = tf.constant([0])

    feat_size = hparams[HP_MEL_BINS.name] * hparams[HP_DOWNSAMPLE_FACTOR.name]

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, feat_size], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        ]
    )
    def greedy_decode(inputs, max_length=None):
        inputs = tf.expand_dims(inputs[0], axis=0)

        encoded = encoder(inputs, training=False)
        enc_length = tf.shape(encoded)[1]

        i_0 = tf.constant(0)
        outputs_0 = tf.expand_dims(start_token, axis=0)
        max_reached_0 = tf.constant(False)

        def time_condition(i, _, max_reached):
            """
            Condition function for the time-step while loop.

            Args:
                i: Current time step index
                outputs: Current output tokens
                max_reached: Boolean indicating if max length was reached

            Returns:
                Boolean indicating whether to continue the loop
            """
            return tf.logical_and(i < enc_length, tf.logical_not(max_reached))

        def time_step_body(i, outputs, max_reached):
            """
            Body function for the time-step while loop.

            Args:
                i: Current time step index
                outputs: Current output tokens
                max_reached: Boolean indicating if max length was reached

            Returns:
                Tuple of (next_index, updated_outputs, updated_max_reached)
            """
            inp_enc = tf.expand_dims(encoded[:, i, :], axis=1)

            _outputs_0 = outputs
            _max_reached_0 = max_reached
            dec_end_0 = tf.constant(False)

            def decoder_condition(_, _max_reached, dec_end):
                """
                Condition function for the decoder while loop.

                Args:
                    _outputs: Current output tokens
                    _max_reached: Boolean indicating if max length was reached
                    dec_end: Boolean indicating if decoding has ended

                Returns:
                    Boolean indicating whether to continue the loop
                """
                return tf.logical_and(
                    tf.logical_not(dec_end), tf.logical_not(_max_reached)
                )

            def decoder_step_body(_outputs, _max_reached, dec_end):
                """
                Body function for the decoder while loop.

                Args:
                    _outputs: Current output tokens
                    _max_reached: Boolean indicating if max length was reached
                    dec_end: Boolean indicating if decoding has ended

                Returns:
                    Tuple of (updated_outputs, updated_max_reached, updated_dec_end)
                """
                pred_out = prediction_network(_outputs, training=False)
                preds = joint(model, inp_enc, pred_out)[0]
                preds = tf.nn.log_softmax(preds)

                predicted_id = tf.cast(tf.argmax(preds, axis=-1), dtype=tf.int32)

                if predicted_id == 0:
                    dec_end = True
                else:
                    _outputs = tf.concat([_outputs, [[predicted_id]]], axis=1)

                if max_length is not None and tf.shape(_outputs)[1] >= max_length + 1:
                    _max_reached = True

                return _outputs, _max_reached, dec_end

            _outputs, _max_reached, _ = tf.while_loop(
                decoder_condition,
                decoder_step_body,
                loop_vars=[_outputs_0, _max_reached_0, dec_end_0],
                shape_invariants=[
                    tf.TensorShape([1, None]),
                    _max_reached_0.get_shape(),
                    dec_end_0.get_shape(),
                ],
            )

            return i + 1, _outputs, _max_reached

        _, outputs, _ = tf.while_loop(
            time_condition,
            time_step_body,
            loop_vars=[i_0, outputs_0, max_reached_0],
            shape_invariants=[
                i_0.get_shape(),
                tf.TensorShape([1, None]),
                max_reached_0.get_shape(),
            ],
        )

        final_outputs = outputs[:, 1:]

        return tf.cast(final_outputs, dtype=tf.int32)

    return greedy_decode
