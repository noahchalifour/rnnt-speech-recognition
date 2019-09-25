import tensorflow as tf
import python_speech_features as psf

def _normalize_text(text):

    return text.lower()


def encode_text(text, vocab):

    norm_text = _normalize_text(text)

    return tf.constant([vocab['<s>']] + [vocab[c] for c in norm_text
                        if c in vocab] + [vocab['</s>']])


def compute_filter_banks(audio_arr):

    return psf.logfbank(audio_arr, winlen=0.025, winstep=0.01,
                        nfilt=80)


def tf_encode_text(text, vocab):

    return tf.py_function(lambda x: encode_text(x.numpy().decode('utf8'), vocab),
        inp=[text], Tout=tf.int32)


def tf_compute_filter_banks(audio_arr):

    return tf.py_function(lambda x: compute_filter_banks(x.numpy()),
        inp=[audio_arr], Tout=tf.float32)


def preprocess_dataset(dataset, vocab, batch_size,
                       shuffle_buffer_size=None):

    _dataset = dataset.map(lambda audio, trans: (
        tf_compute_filter_banks(audio),
        tf_encode_text(trans, vocab),
    ), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.map(lambda audio, labels: (
        audio, labels, tf.shape(audio)[0], tf.shape(labels)[0]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.padded_batch(
        batch_size, padded_shapes=([-1, -1], [-1], [], []))

    _dataset = _dataset.prefetch(
        tf.data.experimental.AUTOTUNE)

    if shuffle_buffer_size is not None:
        _dataset = _dataset.shuffle(shuffle_buffer_size)

    return _dataset