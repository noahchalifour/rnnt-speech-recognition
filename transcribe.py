import tensorflow as tf

try:
    from .utils.data.common import tf_mel_spectrograms, tf_load_audio
except ImportError:
    from utils.data.common import tf_mel_spectrograms, tf_load_audio

def do_inference(model, vocab, audio, sr, max_length=50):

    specs = tf_mel_spectrograms(audio, sr)
    expanded_specs = tf.expand_dims(specs, axis=0)

    output = tf.expand_dims([vocab['<s>']], axis=0)

    @tf.function(input_signature=[tf.TensorSpec([1, None, 80], dtype=tf.float32),
                                  tf.TensorSpec([1, None], dtype=tf.int32),
                                  tf.TensorSpec([2, 1, None], dtype=tf.float32)])
    def infer_step(enc_inp, pred_inp, enc_state):

        predictions, new_enc_state = model([enc_inp, pred_inp, enc_state], 
            training=False)
        predictions = predictions[:, -1:, -1, :]

        return tf.cast(tf.argmax(predictions, axis=-1), tf.int32), new_enc_state

    enc_state = model.initial_state(1)

    for _ in range(max_length):

        predicted_id, enc_state = infer_step(expanded_specs, output, 
            enc_state=enc_state)

        if predicted_id == vocab['</s>']:
            return tf.squeeze(output[:, 1:], axis=0)

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output[:, 1:], axis=0)


def transcribe_file(model, vocab, filepath):

    idx_to_c = tf.constant(list(vocab.keys()), dtype=tf.string)
    audio, sr = tf_load_audio(filepath)
    
    result = do_inference(model, vocab, audio, sr)
    transcript = ''

    for c in result:
        transcript += idx_to_c[c]

    return transcript.numpy().decode('utf8')


def transcribe_stream(model, vocab, stream):

    pass