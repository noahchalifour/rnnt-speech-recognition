import tensorflow as tf

def tf_load_audio(path):

    audio_raw = tf.io.read_file(path)

    return tf.audio.decode_wav(audio_raw)


def _normalize_text(text):

    return text.lower()


def encode_text(text, vocab):

    norm_text = _normalize_text(text)
    encoded = [vocab.index('<s>')]

    for c in norm_text:
        if c == ' ':
            c = '<space>'
        if c in vocab:
            encoded.append(vocab.index(c))
    
    encoded.append(vocab.index('</s>'))

    return tf.constant(encoded)


def tf_encode_text(text, vocab):

    return tf.py_function(lambda x: encode_text(x.numpy().decode('utf8'), vocab),
        inp=[text], Tout=tf.int32)


def tf_mel_spectrograms(audio_arr, sr, 
                        n_mel_bins=80, frame_length=0.050, frame_step=0.0125, 
                        hertz_low=125.0, hertz_high=7600.0):

    _sr = tf.cast(sr, dtype=tf.float32)

    frame_length = tf.cast(tf.round(_sr * frame_length), dtype=tf.int32)
    frame_step = tf.cast(tf.round(_sr * frame_step), dtype=tf.int32)

    stfts = tf.signal.stft(tf.transpose(audio_arr),
                           frame_length=frame_length,
                           frame_step=frame_step)

    mag_specs = tf.abs(stfts)
    num_spec_bins = tf.shape(mag_specs)[-1]

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mel_bins, num_spectrogram_bins=num_spec_bins, 
        # sample_rate=22050.0, 
        sample_rate=_sr,
        lower_edge_hertz=hertz_low,
        upper_edge_hertz=hertz_high)

    mel_specs = tf.tensordot(mag_specs, linear_to_mel_weight_matrix, 1)

    return tf.squeeze(mel_specs)


def preprocess_dataset(dataset, vocab, batch_size,
                       shuffle_buffer_size=None):

    _dataset = dataset.map(lambda audio, sr, trans: (
        tf_mel_spectrograms(audio, sr,
            frame_length=0.025, frame_step=0.01),
        tf_encode_text(trans, vocab),
    ), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.map(lambda audio, labels: (
        audio, labels, tf.shape(audio)[0] - 1, tf.shape(labels)[0] - 2),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.padded_batch(
        batch_size, padded_shapes=([-1, -1], [-1], [], []))

    _dataset = _dataset.prefetch(
        tf.data.experimental.AUTOTUNE)

    if shuffle_buffer_size is not None:
        _dataset = _dataset.shuffle(shuffle_buffer_size)

    return _dataset