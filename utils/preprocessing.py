import glob
import os
import tensorflow as tf

from hparams import *


def tf_load_audio(path):

    audio_raw = tf.io.read_file(path)

    return tf.audio.decode_wav(audio_raw)


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


def downsample_spec(mel_spec, n=3):

    chunk_size = tf.shape(mel_spec)[0] // n
    spec_len = chunk_size * n
    spec_trimmed = mel_spec[:spec_len]

    spec_stack = tf.reshape(spec_trimmed, (chunk_size, n, -1))

    return spec_stack[:, -1, :]


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


def preprocess_dataset(dataset,
                       encoder_fn, 
                       hparams):

    _dataset = dataset.map(lambda audio, sr, trans: (
        compute_mel_spectrograms(audio, sr,
            n_mel_bins=hparams[HP_MEL_BINS.name],
            frame_length=hparams[HP_FRAME_LENGTH.name],
            frame_step=hparams[HP_FRAME_STEP.name],
            hertz_low=hparams[HP_HERTZ_LOW.name],
            hertz_high=hparams[HP_HERTZ_HIGH.name]),
        encoder_fn(trans),
    ), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.map(lambda mel_spec, labels: (
        downsample_spec(mel_spec), labels
    ), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.map(lambda mel_spec, labels: (
            mel_spec, labels, tf.shape(mel_spec)[0], tf.shape(labels)[0],
            labels[1:]
    ), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    _dataset = _dataset.map(tf_serialize_example)

    return _dataset