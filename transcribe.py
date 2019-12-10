import tensorflow as tf

try:
    from .utils.data.common import tf_mel_spectrograms, tf_load_audio
except ImportError:
    from utils.data.common import tf_mel_spectrograms, tf_load_audio

def do_inference(model, audio, sr, max_length=50):

    specs = tf_mel_spectrograms(audio, sr)
    expanded_specs = tf.expand_dims(specs, axis=0)

    output = tf.expand_dims([vocab['<s>']], axis=0)

    enc_state = model.initial_state(1)

    for _ in range(max_length):

        predicted_id, enc_state = model.predict(expanded_specs, output, 
            enc_state=enc_state)

        if predicted_id == vocab['</s>']:
            return tf.squeeze(output[:, 1:], axis=0)

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output[:, 1:], axis=0)


def transcribe_file(model, filepath, max_length=50):

    audio, sr = tf_load_audio(filepath)
    result = ['<s>']
    enc_state = model.initial_state(1)

    audio = tf.squeeze(audio)

    for _ in range(max_length):
    
        pred, enc_state = model.predict([audio], [sr], [result], [enc_state])

        if pred[0] == '</s>':
            break

        result += pred

    return ''.join(result[1:])


def transcribe_stream(model, stream, sr):

    output = ['<s>']
    enc_state = model.initial_state(1)
    _audio = []

    for audio in stream:

        if len(_audio):
            _audio += audio
        else:
            _audio = audio

        pred, enc_state = model.predict([_audio], [sr], [output],
            enc_state=[enc_state])

        if pred[0] == '</s>':

            yield ''.join(output[1:]), True

            output = ['<s>']
            enc_state = model.initial_state(1)
            _audio = []

            continue

        yield ''.join(output[1:]), False

        output += pred