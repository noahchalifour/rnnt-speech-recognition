import warprnnt_tensorflow
import tensorflow as tf

FEAT_SIZE = 80

def rnnt_loss(input_lengths,
              label_lengths):

    def rnnt_loss_fn(y_true, y_pred):

        return warprnnt_tensorflow.rnnt_loss(y_pred, 
            y_true, input_lengths, label_lengths)

    return rnnt_loss_fn

def encoder(num_layers,
            layer_size):

    inputs = tf.keras.layers.Input(shape=(None, FEAT_SIZE))

    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.LSTM(layer_size,
            return_sequences=True)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def prediction_network(vocab_size,
                       embedding_size,
                       num_layers,
                       layer_size):

    inputs = tf.keras.layers.Input(shape=(None,))

    embed = tf.keras.layers.Embedding(vocab_size, embedding_size)(inputs)

    x = embed
    for _ in range(num_layers):
        x = tf.keras.layers.LSTM(layer_size,
            return_sequences=True)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def joint_network(input_shape, 
                  num_units):

    inputs = tf.keras.layers.Input(shape=input_shape)

    outputs = tf.keras.layers.Dense(num_units)(inputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def transducer(vocab_size,
               embedding_size=64,
               encoder_layers=8,
               encoder_size=2048,
               pred_net_layers=2,
               pred_net_size=2048,
               joint_net_size=640,
               softmax_size=4096):

    encoder_inputs = tf.keras.layers.Input(shape=(None, FEAT_SIZE))
    pred_inputs = tf.keras.layers.Input(shape=(None,))
    input_lengths = tf.keras.layers.Input(shape=(None,))
    label_lengths = tf.keras.layers.Input(shape=(None,))

    inputs_enc = encoder(encoder_layers, encoder_size)(encoder_inputs)
    pred_outputs = prediction_network(vocab_size, embedding_size,
        pred_net_layers, pred_net_size)(pred_inputs)

    joint_inputs = (tf.expand_dims(inputs_enc, 2)      # [B, T, V] => [B, T, 1, V]
        + tf.expand_dims(pred_outputs, 1))    # [B, U, V] => [B, 1, U, V]

    joint_outputs = joint_network(joint_inputs.shape, joint_net_size)(joint_inputs)

    soft_out = tf.keras.layers.Dense(softmax_size, activation='softmax')(joint_outputs)
    predictions = tf.keras.layers.Dense(vocab_size, activation='softmax')(soft_out)

    model = tf.keras.Model(inputs=[encoder_inputs, pred_inputs, input_lengths, label_lengths], outputs=predictions)
    model.compile(optimizer='adam', loss=rnnt_loss(input_lengths, label_lengths), metrics=[])

    return model