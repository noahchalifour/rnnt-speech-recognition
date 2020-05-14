import tensorflow as tf

from hparams import *


def joint(model, f, g):

    dense_1 = model.layers[-2]
    dense_2 = model.layers[-1]

    joint_inp = (
        tf.expand_dims(f, axis=2) +                 # [B, T, V] => [B, T, 1, V]
        tf.expand_dims(g[:, -1, :], axis=1))        # [B, U, V] => [B, 1, U, V]

    outputs = dense_1(joint_inp)
    outputs = dense_2(outputs)

    return outputs[:, 0, 0, :]


def greedy_decode_fn(model, hparams):

    # NOTE: Only the first input is decoded

    encoder = model.layers[2]
    prediction_network = model.layers[3]

    start_token = tf.constant([0])

    feat_size = hparams[HP_MEL_BINS.name] * hparams[HP_DOWNSAMPLE_FACTOR.name]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, feat_size], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32)])
    def greedy_decode(inputs, max_length=None):

        inputs = tf.expand_dims(inputs[0], axis=0)

        encoded = encoder(inputs, training=False)
        enc_length = tf.shape(encoded)[1]

        i_0 = tf.constant(0)
        outputs_0 = tf.expand_dims(start_token, axis=0)
        max_reached_0 = tf.constant(False)

        time_cond = lambda i, outputs, max_reached: tf.logical_and(
            i < enc_length, tf.logical_not(max_reached))

        def time_step_body(i, outputs, max_reached):

            inp_enc = tf.expand_dims(encoded[:, i, :],
                axis=1)

            _outputs_0 = outputs
            _max_reached_0 = max_reached
            dec_end_0 = tf.constant(False)

            dec_cond = lambda _outputs, _max_reached, dec_end: tf.logical_and(
                tf.logical_not(dec_end), tf.logical_not(_max_reached))

            def dec_step_body(_outputs, _max_reached, dec_end):

                pred_out = prediction_network(_outputs,
                    training=False)
                preds = joint(model, inp_enc, pred_out)[0]
                preds = tf.nn.log_softmax(preds)

                predicted_id = tf.cast(
                    tf.argmax(preds, axis=-1), dtype=tf.int32)

                if predicted_id == 0:
                    dec_end = True
                else:
                    _outputs = tf.concat([_outputs, [[predicted_id]]],
                        axis=1)

                if max_length is not None and tf.shape(_outputs)[1] >= max_length + 1:
                    _max_reached = True

                return _outputs, _max_reached, dec_end

            _outputs, _max_reached, _ = tf.while_loop(
                dec_cond, dec_step_body,
                loop_vars=[_outputs_0, _max_reached_0, dec_end_0],
                shape_invariants=[
                    tf.TensorShape([1, None]),
                    _max_reached_0.get_shape(),
                    dec_end_0.get_shape()
                ])

            return i + 1, _outputs, _max_reached

        _, outputs, _ = tf.while_loop(
            time_cond, time_step_body,
            loop_vars=[i_0, outputs_0, max_reached_0],
            shape_invariants=[
                i_0.get_shape(),
                tf.TensorShape([1, None]),
                max_reached_0.get_shape()
            ])

        final_outputs = outputs[:, 1:]
        # output_ids = tf.argmax(final_outputs, axis=-1)

        # return tf.cast(output_ids, dtype=tf.int32)
        return tf.cast(final_outputs, dtype=tf.int32)

    return greedy_decode


# def greedy_decode():

#     # NOTE: Only the first input is decoded
#     y_pred = y_pred[0]

#     # Add blank at end for decoding
#     pred_len = tf.shape(y_pred)[0]
#     y_pred = tf.concat([y_pred,
#                         tf.fill([pred_len, 1], 0)],
#         axis=1)

#     def _loop_body(_y_pred, _decoded):

#         first_blank_idx = tf.cast(tf.where(
#             tf.equal(_y_pred[0], 0)), dtype=tf.int32)
#         has_blank = tf.not_equal(tf.size(first_blank_idx), 0)

#         dec_idx = first_blank_idx[0][0]

#         decoded = _y_pred[0][:dec_idx]
#         n_dec = tf.shape(decoded)[0]

#         _decoded = tf.concat([_decoded, decoded],
#             axis=0)

#         return _y_pred[1:, n_dec:], _decoded

#     decoded_0 = tf.constant([], dtype=tf.int32)

#     _, decoded = tf.while_loop(
#         lambda _y_pred, _decoded: tf.not_equal(tf.size(_y_pred), 0),
#         _loop_body,
#         [y_pred, decoded_0],
#         shape_invariants=[tf.TensorShape([None, None]), tf.TensorShape([None])],
#         name='greedy_decode')

#     return tf.expand_dims(decoded, axis=0)


# a = tf.constant([
#     [
#         [1, 4, 4, 4, 4, 3, 2],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 4, 1, 4, 0]
#     ]
# ])

# tf.config.experimental_run_functions_eagerly(True)

# a = tf.zeros((4, 100, 80))

# print(a)

# import sys
# import os

# FILE_DIR = os.path.dirname(os.path.realpath(__file__))
# sys.path = [os.path.join(FILE_DIR, '..')] + sys.path

# from model import build_keras_model
# from hparams import *

# hparams = {

#     HP_TOKEN_TYPE: HP_TOKEN_TYPE.domain.values[1],

#     # Preprocessing
#     HP_MEL_BINS: HP_MEL_BINS.domain.values[0],
#     HP_FRAME_LENGTH: HP_FRAME_LENGTH.domain.values[0],
#     HP_FRAME_STEP: HP_FRAME_STEP.domain.values[0],
#     HP_HERTZ_LOW: HP_HERTZ_LOW.domain.values[0],
#     HP_HERTZ_HIGH: HP_HERTZ_HIGH.domain.values[0],

#     # Model
#     HP_EMBEDDING_SIZE: HP_EMBEDDING_SIZE.domain.values[0],
#     HP_ENCODER_LAYERS: HP_ENCODER_LAYERS.domain.values[0],
#     HP_ENCODER_SIZE: HP_ENCODER_SIZE.domain.values[0],
#     HP_PROJECTION_SIZE: HP_PROJECTION_SIZE.domain.values[0],
#     HP_TIME_REDUCT_INDEX: HP_TIME_REDUCT_INDEX.domain.values[0],
#     HP_TIME_REDUCT_FACTOR: HP_TIME_REDUCT_FACTOR.domain.values[0],
#     HP_PRED_NET_LAYERS: HP_PRED_NET_LAYERS.domain.values[0],
#     HP_PRED_NET_SIZE: HP_PRED_NET_SIZE.domain.values[0],
#     HP_JOINT_NET_SIZE: HP_JOINT_NET_SIZE.domain.values[0],

#     HP_LEARNING_RATE: HP_LEARNING_RATE.domain.values[0]

# }

# hparams = {k.name: v for k, v in hparams.items()}
# hparams['vocab_size'] = 73

# model = build_keras_model(hparams)

# greedy_decode = greedy_decode_fn(model)

# print(greedy_decode(a, max_length=20))
