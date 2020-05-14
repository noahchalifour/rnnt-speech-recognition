import glob
import os
import librosa.display
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from hparams import *


def tf_load_audio(path, pre_emphasis=0.97):

    audio_raw = tf.io.read_file(path)

    audio, sr = tf.audio.decode_wav(audio_raw)

    if tf.rank(audio) > 1:
        audio = audio[:, 0]

    return audio, sr


def normalize_text(text):

    text = text.lower()
    text = text.replace('"', '')

    return text


def tf_normalize_text(text):

    return tf.py_function(
        lambda x: normalize_text(x.numpy().decode('utf8')),
        inp=[text],
        Tout=tf.string)


def print_tensor(t, template='{}'):

    return tf.py_function(
        lambda x: print(template.format(x.numpy())),
        inp=[t],
        Tout=[])


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

    stfts = tf.signal.stft(audio_arr,
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
    mel_specs.set_shape(mag_specs.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_specs = tf.math.log(mel_specs + 1e-6)
    log_mel_specs -= (tf.reduce_mean(log_mel_specs, axis=0) + 1e-8)

    return log_mel_specs


def downsample_spec(mel_spec, n=3):

    spec_shape = tf.shape(mel_spec)
    spec_length, feat_size = spec_shape[0], spec_shape[1]

    trimmed_length = (spec_length // n) * n

    trimmed_spec = mel_spec[:trimmed_length]
    spec_sampled = tf.reshape(trimmed_spec, (-1, feat_size * n))

    return spec_sampled


def load_dataset(data_dir, name):

    filenames = glob.glob(os.path.join(data_dir,
        '{}.tfrecord'.format(name)))

    raw_dataset = tf.data.TFRecordDataset(filenames)

    parsed_dataset = raw_dataset.map(parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return parsed_dataset


def parse_example(serialized_example):

    parse_dict = {
        'mel_specs': tf.io.FixedLenFeature([], tf.string),
        'pred_inp': tf.io.FixedLenFeature([], tf.string),
        'spec_lengths': tf.io.FixedLenFeature([], tf.string),
        'label_lengths': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(serialized_example, parse_dict)

    mel_specs = tf.io.parse_tensor(example['mel_specs'], out_type=tf.float32)
    pred_inp = tf.io.parse_tensor(example['pred_inp'], out_type=tf.int32)
    spec_lengths = tf.io.parse_tensor(example['spec_lengths'], out_type=tf.int32)
    label_lengths = tf.io.parse_tensor(example['label_lengths'], out_type=tf.int32)

    labels = tf.io.parse_tensor(example['labels'], out_type=tf.int32)

    return (mel_specs, pred_inp, spec_lengths, label_lengths, labels)


def serialize_example(mel_specs,
                      pred_inp,
                      spec_lengths,
                      label_lengths,
                      labels):

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))): # if value ist tensor
            value = value.numpy() # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    mel_specs_s = tf.io.serialize_tensor(mel_specs)
    pred_inp_s = tf.io.serialize_tensor(pred_inp)
    spec_lengths_s = tf.io.serialize_tensor(spec_lengths)
    label_lengths_s = tf.io.serialize_tensor(label_lengths)

    labels_s = tf.io.serialize_tensor(labels)

    feature = {
        'mel_specs': _bytes_feature(mel_specs_s),
        'pred_inp': _bytes_feature(pred_inp_s),
        'spec_lengths': _bytes_feature(spec_lengths_s),
        'label_lengths': _bytes_feature(label_lengths_s),
        'labels': _bytes_feature(labels_s)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()

def tf_serialize_example(mel_specs,
                         pred_inp,
                         spec_lengths,
                         label_lengths,
                         labels):

    tf_string = tf.py_function(
        serialize_example,
        (mel_specs, pred_inp, spec_lengths, label_lengths, labels),
        tf.string)

    return tf.reshape(tf_string, ())


def preprocess_text(text, encoder_fn, vocab_size):

    norm_text = tf_normalize_text(text)
    enc_text = encoder_fn(norm_text)
    enc_padded = tf.concat([[0], enc_text], axis=0)

    return enc_text, enc_padded


def plot_spec(spec, sr, transcription, name):

    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(12,4))
    librosa.display.specshow(spec_db, sr=sr,
        x_axis='time', y_axis='mel',
        hop_length=sr * 0.01)
    plt.colorbar(format='%+02.0f dB')
    plt.savefig('figs/{}.png'.format(name))
    plt.clf()


def tf_plot_spec(spec, sr, transcription, name):

    spec_t = tf.transpose(spec)

    return tf.py_function(
        lambda _spec, _sr, trans: plot_spec(
            _spec.numpy(), _sr.numpy(),
            trans.numpy().decode('utf8'),
            name
        ),
        inp=[spec_t, sr, transcription],
        Tout=[])


def plot_audio(audio_arr, sr, trans, name):

    with open('figs/trans.txt', 'a') as f:
        f.write('{} {}\n'.format(name, trans))

    t = np.linspace(0, audio_arr.shape[0] / sr,
        num=audio_arr.shape[0])

    plt.figure(1)
    plt.plot(t, audio_arr)
    plt.savefig('figs/{}.png'.format(name))
    plt.clf()


def tf_plot_audio(audio_arr, sr, trans, name):

    return tf.py_function(
        lambda _audio, _sr, _trans: plot_audio(
            _audio.numpy(), _sr.numpy(),
            _trans.numpy(), name
        ),
        inp=[audio_arr, sr, trans],
        Tout=[])


def preprocess_audio(audio,
                     sample_rate,
                     hparams):

    log_melspec = compute_mel_spectrograms(
        audio_arr=audio,
        sample_rate=sample_rate,
        n_mel_bins=hparams[HP_MEL_BINS.name],
        frame_length=hparams[HP_FRAME_LENGTH.name],
        frame_step=hparams[HP_FRAME_STEP.name],
        hertz_low=hparams[HP_HERTZ_LOW.name],
        hertz_high=hparams[HP_HERTZ_HIGH.name])

    downsampled_spec = downsample_spec(log_melspec)

    return downsampled_spec


def preprocess_dataset(dataset,
                       encoder_fn,
                       hparams,
                       max_length=0,
                       save_plots=False):

    _dataset = dataset

    if max_length > 0:
        _dataset = _dataset.filter(lambda audio, sr, trans: (
            tf.shape(audio)[0] <= sr * tf.constant(max_length)))

    if save_plots:
        os.makedirs('figs', exist_ok=True)
        for i, (audio_arr, sr, trans) in enumerate(_dataset.take(5)):
            tf_plot_audio(audio_arr, sr, trans, 'audio_{}'.format(i))

    _dataset = _dataset.map(lambda audio, sr, trans: (
        preprocess_audio(
            audio=audio,
            sample_rate=sr,
            hparams=hparams),
        sr,
        *preprocess_text(trans,
            encoder_fn=encoder_fn,
            vocab_size=hparams[HP_VOCAB_SIZE.name]),
        trans
    ), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if save_plots:
        for i, (log_melspec, sr, _, _, trans) in enumerate(_dataset.take(5)):
            tf_plot_spec(log_melspec, sr, trans, 'input_{}'.format(i))

    _dataset = _dataset.map(
        lambda log_melspec, sr, labels, pred_inp, trans: (
            log_melspec, pred_inp,
            tf.shape(log_melspec)[0], tf.shape(labels)[0],
            labels
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.map(tf_serialize_example)

    return _dataset
