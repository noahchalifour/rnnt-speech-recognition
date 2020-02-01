import tensorflow as tf

from hparams import *


def tf_load_audio(path):

    audio_raw = tf.io.read_file(path)

    return tf.audio.decode_wav(audio_raw)


def build_lookup_table(keys, values=None, default_value=-1):

    if values is None:
        values = tf.range(len(keys))

    kv_init = tf.lookup.KeyValueTensorInitializer(
        keys=keys, values=values)

    return tf.lookup.StaticHashTable(kv_init,
        default_value=default_value)


def normalize_text(text):

    return text.lower()


def encode_text(text, vocab_table):

    byte_list = tf.strings.bytes_split(text)
    byte_list = tf.concat([['<s>'], byte_list, ['</s>']], axis=0)

    return vocab_table.lookup(byte_list)


def compute_mel_spectrograms(audio_arr, 
                             sample_rate, 
                             n_mel_bins, 
                             frame_length,
                             frame_step, 
                             hertz_low,
                             hertz_high):

    sample_rate_f = tf.cast(sample_rate, dtype=tf.float32)

    frame_length = tf.cast(tf.round(sample_rate_f * frame_length), dtype=tf.int32)
    frame_step = tf.cast(tf.round(sample_rate_f * frame_step), dtype=tf.int32)

    stfts = tf.signal.stft(tf.transpose(audio_arr),
                           frame_length=frame_length,
                           frame_step=frame_step)

    mag_specs = tf.abs(stfts)
    num_spec_bins = tf.shape(mag_specs)[-1]

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mel_bins, num_spectrogram_bins=num_spec_bins, 
        sample_rate=sample_rate_f,
        lower_edge_hertz=hertz_low,
        upper_edge_hertz=hertz_high)

    mel_specs = tf.tensordot(mag_specs, linear_to_mel_weight_matrix, 1)

    return tf.squeeze(mel_specs)


def preprocess_dataset(dataset, 
                       vocab_table, 
                       batch_size, 
                       hparams):

    _dataset = dataset.shuffle(5000)

    _dataset = _dataset.map(lambda audio, sr, trans: (
        compute_mel_spectrograms(audio, sr,
            n_mel_bins=hparams[HP_MEL_BINS],
            frame_length=hparams[HP_FRAME_LENGTH],
            frame_step=hparams[HP_FRAME_STEP],
            hertz_low=hparams[HP_HERTZ_LOW],
            hertz_high=hparams[HP_HERTZ_HIGH]),
        encode_text(trans, vocab_table),
    ), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.map(lambda audio, labels: ({
            'mel_specs': audio,
            'pred_inp': labels[:-1],
            'spec_lengths': tf.shape(audio)[0] - 1,
            'label_lengths': tf.shape(labels)[0] - 2,
        }, labels[1:]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.padded_batch(
        batch_size, padded_shapes=({
            'mel_specs': [-1, -1], 
            'pred_inp': [-1],
            'spec_lengths': [],
            'label_lengths': []
        }, [-1]))

    enc_state = tf.zeros((2, batch_size, hparams[HP_ENCODER_SIZE]))
    _dataset = _dataset.map(lambda inp, out: ({
        **inp,
        'enc_state': enc_state
    }, out))

    _dataset = _dataset.prefetch(
        tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.repeat()

    return _dataset