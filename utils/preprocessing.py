"""
Audio and text preprocessing utilities for speech recognition.

This module provides functions for processing audio and text data for speech recognition tasks,
including loading audio files, computing mel spectrograms, normalizing text, and preparing
data for training RNN-T models. It also includes utilities for serializing and deserializing
TFRecord datasets and visualization functions for debugging.
"""

import glob
import os
import librosa.display
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from hparams import (
    HP_MEL_BINS,
    HP_FRAME_LENGTH,
    HP_FRAME_STEP,
    HP_HERTZ_LOW,
    HP_HERTZ_HIGH,
)


def tf_load_audio(path):
    """
    Load an audio file using TensorFlow operations.

    This function reads an audio file from the given path and decodes it as a WAV file.
    If the audio has multiple channels, only the first channel is returned.

    Args:
        path (tf.Tensor): A string tensor containing the path to the audio file.

    Returns:
        tuple: A tuple containing:
            - audio (tf.Tensor): The audio samples as a 1D tensor.
            - sr (tf.Tensor): The sample rate of the audio.
    """
    audio_raw = tf.io.read_file(path)

    audio, sr = tf.audio.decode_wav(audio_raw)

    if tf.rank(audio) > 1:
        audio = audio[:, 0]

    return audio, sr


def normalize_text(text):
    """
    Normalize text for speech recognition.

    This function performs basic text normalization by converting text to lowercase
    and removing double quotes.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    text = text.lower()
    text = text.replace('"', "")

    return text


def _normalize_text_py_func(x):
    """
    Helper function for tf_normalize_text to be used with tf.py_function.

    Args:
        x (tf.Tensor): A string tensor to normalize.

    Returns:
        str: The normalized text.
    """
    return normalize_text(x.numpy().decode("utf8"))


def tf_normalize_text(text):
    """
    TensorFlow wrapper for the normalize_text function.

    This function wraps the normalize_text function to be used within TensorFlow operations.
    It decodes the input tensor to UTF-8 string, applies text normalization, and returns
    the result as a TensorFlow string tensor.

    Args:
        text (tf.Tensor): A string tensor to normalize.

    Returns:
        tf.Tensor: The normalized text as a string tensor.
    """
    return tf.py_function(_normalize_text_py_func, inp=[text], Tout=tf.string)


def _print_tensor_py_func(x, template):
    """
    Helper function for print_tensor to be used with tf.py_function.

    Args:
        x (tf.Tensor): The tensor to print.
        template (str): A format string template for printing.

    Returns:
        None
    """
    print(template.format(x.numpy()))


def print_tensor(t, template="{}"):
    """
    Print a tensor's value during TensorFlow execution.

    This function is useful for debugging TensorFlow operations by printing tensor values
    during graph execution. It uses tf.py_function to execute a Python print statement
    within the TensorFlow graph.

    Args:
        t (tf.Tensor): The tensor to print.
        template (str, optional): A format string template for printing. Defaults to '{}'.
            The tensor value will be inserted into this template.

    Returns:
        tf.Operation: A TensorFlow operation that, when executed, prints the tensor value.
    """

    # Create a closure that captures the template parameter
    def _print_wrapper(x):
        return _print_tensor_py_func(x, template)

    return tf.py_function(_print_wrapper, inp=[t], Tout=[])


def compute_mel_spectrograms(
    audio_arr, sample_rate, n_mel_bins, frame_length, frame_step, hertz_low, hertz_high
):
    """
    Compute log mel spectrograms from audio data using TensorFlow operations.

    This function computes the Short-Time Fourier Transform (STFT) of the input audio,
    converts it to mel-scale spectrograms, and applies logarithmic scaling with
    mean normalization.

    Args:
        audio_arr (tf.Tensor): Audio samples as a 1D tensor.
        sample_rate (tf.Tensor): Sample rate of the audio.
        n_mel_bins (int): Number of mel frequency bins to generate.
        frame_length (float): Frame length in seconds.
        frame_step (float): Frame step (hop length) in seconds.
        hertz_low (float): Lowest frequency in Hz for mel filterbank.
        hertz_high (float): Highest frequency in Hz for mel filterbank.

    Returns:
        tf.Tensor: Log mel spectrograms with shape [time_steps, n_mel_bins].
    """
    sample_rate_f = tf.cast(sample_rate, dtype=tf.float32)

    frame_length = tf.cast(tf.round(sample_rate_f * frame_length), dtype=tf.int32)
    frame_step = tf.cast(tf.round(sample_rate_f * frame_step), dtype=tf.int32)

    stfts = tf.signal.stft(audio_arr, frame_length=frame_length, frame_step=frame_step)

    mag_specs = tf.abs(stfts)
    num_spec_bins = tf.shape(mag_specs)[-1]

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mel_bins,
        num_spectrogram_bins=num_spec_bins,
        sample_rate=sample_rate_f,
        lower_edge_hertz=hertz_low,
        upper_edge_hertz=hertz_high,
    )

    mel_specs = tf.tensordot(mag_specs, linear_to_mel_weight_matrix, 1)
    mel_specs.set_shape(
        mag_specs.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )

    log_mel_specs = tf.math.log(mel_specs + 1e-6)
    log_mel_specs -= tf.reduce_mean(log_mel_specs, axis=0) + 1e-8

    return log_mel_specs


def downsample_spec(mel_spec, n=3):
    """
    Downsample a mel spectrogram by reshaping consecutive frames.

    This function takes a mel spectrogram and downsamples it by combining
    consecutive frames. It first ensures the spectrogram length is divisible
    by n by trimming excess frames, then reshapes the spectrogram to combine
    every n consecutive frames into a single frame with n times the features.

    Args:
        mel_spec (tf.Tensor): Input mel spectrogram with shape [time_steps, features].
        n (int, optional): Number of consecutive frames to combine. Defaults to 3.

    Returns:
        tf.Tensor: Downsampled spectrogram with shape [time_steps/n, features*n].
    """
    spec_shape = tf.shape(mel_spec)
    spec_length, feat_size = spec_shape[0], spec_shape[1]

    trimmed_length = (spec_length // n) * n

    trimmed_spec = mel_spec[:trimmed_length]
    spec_sampled = tf.reshape(trimmed_spec, (-1, feat_size * n))

    return spec_sampled


def load_dataset(data_dir, name):
    """
    Load a TFRecord dataset from disk.

    This function loads a TFRecord dataset from the specified directory and name,
    parses the examples using the parse_example function, and returns a TensorFlow dataset.

    Args:
        data_dir (str): Directory containing the TFRecord files.
        name (str): Base name of the TFRecord files (without the .tfrecord extension).

    Returns:
        tf.data.Dataset: A parsed TensorFlow dataset ready for training or evaluation.
    """
    filenames = glob.glob(os.path.join(data_dir, "{}.tfrecord".format(name)))

    raw_dataset = tf.data.TFRecordDataset(filenames)

    parsed_dataset = raw_dataset.map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return parsed_dataset


def parse_example(serialized_example):
    """
    Parse a serialized TFRecord example for speech recognition.

    This function deserializes a TFRecord example containing mel spectrograms,
    prediction inputs, spectrogram lengths, label lengths, and labels for
    speech recognition training.

    Args:
        serialized_example (tf.Tensor): A serialized TFRecord example.

    Returns:
        tuple: A tuple containing:
            - mel_specs (tf.Tensor): Mel spectrograms as a float32 tensor.
            - pred_inp (tf.Tensor): Prediction inputs as an int32 tensor.
            - spec_lengths (tf.Tensor): Spectrogram lengths as an int32 tensor.
            - label_lengths (tf.Tensor): Label lengths as an int32 tensor.
            - labels (tf.Tensor): Labels as an int32 tensor.
    """
    parse_dict = {
        "mel_specs": tf.io.FixedLenFeature([], tf.string),
        "pred_inp": tf.io.FixedLenFeature([], tf.string),
        "spec_lengths": tf.io.FixedLenFeature([], tf.string),
        "label_lengths": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(serialized_example, parse_dict)

    mel_specs = tf.io.parse_tensor(example["mel_specs"], out_type=tf.float32)
    pred_inp = tf.io.parse_tensor(example["pred_inp"], out_type=tf.int32)
    spec_lengths = tf.io.parse_tensor(example["spec_lengths"], out_type=tf.int32)
    label_lengths = tf.io.parse_tensor(example["label_lengths"], out_type=tf.int32)

    labels = tf.io.parse_tensor(example["labels"], out_type=tf.int32)

    return (mel_specs, pred_inp, spec_lengths, label_lengths, labels)


def serialize_example(mel_specs, pred_inp, spec_lengths, label_lengths, labels):
    """
    Serialize speech recognition data into a TFRecord example.

    This function takes preprocessed speech recognition data (mel spectrograms,
    prediction inputs, lengths, and labels) and serializes them into a TFRecord
    example format for efficient storage and loading.

    Args:
        mel_specs (tf.Tensor): Mel spectrograms as a float32 tensor.
        pred_inp (tf.Tensor): Prediction inputs as an int32 tensor.
        spec_lengths (tf.Tensor): Spectrogram lengths as an int32 tensor.
        label_lengths (tf.Tensor): Label lengths as an int32 tensor.
        labels (tf.Tensor): Labels as an int32 tensor.

    Returns:
        bytes: A serialized TFRecord example containing the input data.
    """

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):  # if value ist tensor
            value = value.numpy()  # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    mel_specs_s = tf.io.serialize_tensor(mel_specs)
    pred_inp_s = tf.io.serialize_tensor(pred_inp)
    spec_lengths_s = tf.io.serialize_tensor(spec_lengths)
    label_lengths_s = tf.io.serialize_tensor(label_lengths)

    labels_s = tf.io.serialize_tensor(labels)

    feature = {
        "mel_specs": _bytes_feature(mel_specs_s),
        "pred_inp": _bytes_feature(pred_inp_s),
        "spec_lengths": _bytes_feature(spec_lengths_s),
        "label_lengths": _bytes_feature(label_lengths_s),
        "labels": _bytes_feature(labels_s),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


def tf_serialize_example(mel_specs, pred_inp, spec_lengths, label_lengths, labels):
    """
    TensorFlow wrapper for the serialize_example function.

    This function wraps the serialize_example function to be used within TensorFlow operations.
    It takes preprocessed speech recognition data and serializes them into a TFRecord example
    format that can be used in a TensorFlow data pipeline.

    Args:
        mel_specs (tf.Tensor): Mel spectrograms as a float32 tensor.
        pred_inp (tf.Tensor): Prediction inputs as an int32 tensor.
        spec_lengths (tf.Tensor): Spectrogram lengths as an int32 tensor.
        label_lengths (tf.Tensor): Label lengths as an int32 tensor.
        labels (tf.Tensor): Labels as an int32 tensor.

    Returns:
        tf.Tensor: A scalar string tensor containing the serialized TFRecord example.
    """
    tf_string = tf.py_function(
        serialize_example,
        (mel_specs, pred_inp, spec_lengths, label_lengths, labels),
        tf.string,
    )

    return tf.reshape(tf_string, ())


def preprocess_text(text, encoder_fn):
    """
    Preprocess text for speech recognition model input.

    This function normalizes the input text and encodes it using the provided encoder function.
    It also creates a padded version of the encoded text by prepending a 0 token, which is
    typically used as the blank/start token for RNN-T prediction.

    Args:
        text (tf.Tensor): A string tensor containing the text to preprocess.
        encoder_fn (callable): A function that encodes normalized text into integer tokens.

    Returns:
        tuple: A tuple containing:
            - enc_text (tf.Tensor): The encoded text as an int tensor.
            - enc_padded (tf.Tensor): The encoded text with a 0 token prepended.
    """
    norm_text = tf_normalize_text(text)
    enc_text = encoder_fn(norm_text)
    enc_padded = tf.concat([[0], enc_text], axis=0)

    return enc_text, enc_padded


def plot_spec(spec, sr, name):
    """
    Plot a mel spectrogram and save it as an image.

    This function converts a mel spectrogram to decibel scale and creates a visualization
    using librosa's display functions. The resulting plot is saved as a PNG file in the
    'figs' directory.

    Args:
        spec (numpy.ndarray): The mel spectrogram to plot.
        sr (int): Sample rate of the audio.
        name (str): Base name for the output file (without extension).

    Returns:
        None: The function saves the plot to a file but doesn't return a value.
    """
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        spec_db, sr=sr, x_axis="time", y_axis="mel", hop_length=sr * 0.01
    )
    plt.colorbar(format="%+02.0f dB")
    plt.savefig("figs/{}.png".format(name))
    plt.clf()


def _plot_spec_py_func(_spec, _sr, name):
    """
    Helper function for tf_plot_spec to be used with tf.py_function.

    Args:
        _spec (tf.Tensor): The transposed mel spectrogram to plot.
        _sr (tf.Tensor): Sample rate of the audio.
        name (str): Base name for the output file (without extension).

    Returns:
        None
    """
    plot_spec(_spec.numpy(), _sr.numpy(), name)


def tf_plot_spec(spec, sr, name):
    """
    TensorFlow wrapper for the plot_spec function.

    This function wraps the plot_spec function to be used within TensorFlow operations.
    It transposes the input spectrogram and calls plot_spec to create and save a
    visualization of the mel spectrogram.

    Args:
        spec (tf.Tensor): The mel spectrogram to plot.
        sr (tf.Tensor): Sample rate of the audio.
        name (str): Base name for the output file (without extension).

    Returns:
        tf.Operation: A TensorFlow operation that, when executed, creates and saves the plot.
    """
    spec_t = tf.transpose(spec)

    # Create a closure that captures the name parameter
    def _plot_spec_wrapper(_spec, _sr):
        return _plot_spec_py_func(_spec, _sr, name)

    return tf.py_function(_plot_spec_wrapper, inp=[spec_t, sr], Tout=[])


def plot_audio(audio_arr, sr, trans, name):
    """
    Plot audio waveform and save it as an image.

    This function creates a visualization of an audio waveform and saves it as a PNG file
    in the 'figs' directory. It also appends the transcription text to a file named
    'trans.txt' in the same directory.

    Args:
        audio_arr (numpy.ndarray): The audio samples to plot.
        sr (int): Sample rate of the audio.
        name (str): Base name for the output file (without extension).

    Returns:
        None: The function saves the plot to a file but doesn't return a value.
    """
    with open("figs/trans.txt", "a") as f:
        f.write("{} {}\n".format(name, trans))

    t = np.linspace(0, audio_arr.shape[0] / sr, num=audio_arr.shape[0])

    plt.figure(1)
    plt.plot(t, audio_arr)
    plt.savefig("figs/{}.png".format(name))
    plt.clf()


def _plot_audio_py_func(_audio, _sr, _trans, name):
    """
    Helper function for tf_plot_audio to be used with tf.py_function.

    Args:
        _audio (tf.Tensor): The audio samples to plot.
        _sr (tf.Tensor): Sample rate of the audio.
        _trans (tf.Tensor): Transcription text for the audio.
        name (str): Base name for the output file (without extension).

    Returns:
        None
    """
    plot_audio(_audio.numpy(), _sr.numpy(), _trans.numpy(), name)


def tf_plot_audio(audio_arr, sr, trans, name):
    """
    TensorFlow wrapper for the plot_audio function.

    This function wraps the plot_audio function to be used within TensorFlow operations.
    It calls plot_audio to create and save a visualization of the audio waveform.

    Args:
        audio_arr (tf.Tensor): The audio samples to plot.
        sr (tf.Tensor): Sample rate of the audio.
        trans (tf.Tensor): Transcription text for the audio.
        name (str): Base name for the output file (without extension).

    Returns:
        tf.Operation: A TensorFlow operation that, when executed, creates and saves the plot.
    """

    # Create a closure that captures the name parameter
    def _plot_audio_wrapper(_audio, _sr, _trans):
        return _plot_audio_py_func(_audio, _sr, _trans, name)

    return tf.py_function(_plot_audio_wrapper, inp=[audio_arr, sr, trans], Tout=[])


def preprocess_audio(audio, sample_rate, hparams):
    """
    Preprocess audio data for speech recognition model input.

    This function converts raw audio data into a format suitable for speech recognition
    models by computing log mel spectrograms and then downsampling them. It uses
    hyperparameters from the hparams dictionary to configure the processing.

    Args:
        audio (tf.Tensor): Raw audio samples as a 1D tensor.
        sample_rate (tf.Tensor): Sample rate of the audio.
        hparams (dict): Dictionary of hyperparameters containing:
            - HP_MEL_BINS.name: Number of mel frequency bins
            - HP_FRAME_LENGTH.name: Frame length in seconds
            - HP_FRAME_STEP.name: Frame step (hop length) in seconds
            - HP_HERTZ_LOW.name: Lowest frequency in Hz for mel filterbank
            - HP_HERTZ_HIGH.name: Highest frequency in Hz for mel filterbank

    Returns:
        tf.Tensor: Downsampled log mel spectrogram ready for model input.
    """
    log_melspec = compute_mel_spectrograms(
        audio_arr=audio,
        sample_rate=sample_rate,
        n_mel_bins=hparams[HP_MEL_BINS.name],
        frame_length=hparams[HP_FRAME_LENGTH.name],
        frame_step=hparams[HP_FRAME_STEP.name],
        hertz_low=hparams[HP_HERTZ_LOW.name],
        hertz_high=hparams[HP_HERTZ_HIGH.name],
    )

    downsampled_spec = downsample_spec(log_melspec)

    return downsampled_spec


def tf_filter_by_length(audio, sr, _, max_length):
    """
    Filter function to exclude audio samples longer than max_length.

    Args:
        audio (tf.Tensor): Audio samples as a 1D tensor.
        sr (tf.Tensor): Sample rate of the audio.
        _ (tf.Tensor): Unused parameter (transcription).
        max_length (float): Maximum audio length in seconds.

    Returns:
        tf.Tensor: Boolean tensor indicating whether to keep the sample.
    """
    return tf.shape(audio)[0] <= sr * tf.constant(max_length)


def preprocess_dataset(dataset, encoder_fn, hparams, max_length=0, save_plots=False):
    """
    Preprocess a dataset of audio samples for speech recognition training.

    This function takes a dataset containing audio samples, sample rates, and transcriptions,
    and processes it into a format suitable for training an RNN-T speech recognition model.
    The processing includes:
    1. Filtering out audio samples longer than max_length (if specified)
    2. Optionally saving plots of audio waveforms for visualization
    3. Converting audio to mel spectrograms and downsampling
    4. Preprocessing and encoding text transcriptions
    5. Extracting necessary features and lengths
    6. Serializing the processed data into TFRecord format

    Args:
        dataset (tf.data.Dataset): Input dataset with elements (audio, sample_rate, transcription).
        encoder_fn (callable): Function to encode text into integer tokens.
        hparams (dict): Dictionary of hyperparameters for audio and text processing.
        max_length (float, optional): Maximum audio length in seconds. Samples longer than
            this will be filtered out. If 0, no filtering is applied. Defaults to 0.
        save_plots (bool, optional): Whether to save visualization plots of audio and
            spectrograms for debugging. Defaults to False.

    Returns:
        tf.data.Dataset: A dataset of serialized TFRecord examples ready for training.
    """
    _dataset = dataset

    if max_length > 0:

        def _filter_by_length(audio, sr, _):
            return tf_filter_by_length(audio, sr, _, max_length)

        _dataset = _dataset.filter(_filter_by_length)

    if save_plots:
        os.makedirs("figs", exist_ok=True)
        for i, (audio_arr, sr, trans) in enumerate(_dataset.take(5)):
            tf_plot_audio(audio_arr, sr, trans, "audio_{}".format(i))

    def _preprocess_audio_and_text(audio, sr, trans, hparams, encoder_fn):
        """
        Preprocess audio and text data for speech recognition.

        Args:
            audio (tf.Tensor): Audio samples as a 1D tensor.
            sr (tf.Tensor): Sample rate of the audio.
            trans (tf.Tensor): Transcription text for the audio.
            hparams (dict): Dictionary of hyperparameters for audio processing.
            encoder_fn (callable): Function to encode normalized text into integer tokens.

        Returns:
            tuple: A tuple containing processed audio, sample rate, encoded text, padded text, and original transcription.
        """
        processed_audio = preprocess_audio(audio=audio, sample_rate=sr, hparams=hparams)
        encoded_text, padded_text = preprocess_text(trans, encoder_fn=encoder_fn)
        return processed_audio, sr, encoded_text, padded_text, trans

    # Create a closure that captures the hparams and encoder_fn parameters
    def _preprocess_wrapper(audio, sr, trans):
        return _preprocess_audio_and_text(audio, sr, trans, hparams, encoder_fn)

    _dataset = _dataset.map(
        _preprocess_wrapper,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if save_plots:
        for i, (log_melspec, sr, _, _, _) in enumerate(_dataset.take(5)):
            tf_plot_spec(log_melspec, sr, "input_{}".format(i))

    def _extract_features_and_lengths(log_melspec, _, labels, pred_inp, trans):
        """
        Extract features and lengths for model training.

        Args:
            log_melspec (tf.Tensor): Log mel spectrogram.
            _ (tf.Tensor): Unused parameter (sample rate).
            labels (tf.Tensor): Encoded text labels.
            pred_inp (tf.Tensor): Prediction inputs.
            trans (tf.Tensor): Transcription text.

        Returns:
            tuple: A tuple containing log mel spectrogram, prediction inputs, spectrogram length, label length, and labels.
        """
        return (
            log_melspec,
            pred_inp,
            tf.shape(log_melspec)[0],
            tf.shape(labels)[0],
            labels,
        )

    _dataset = _dataset.map(
        _extract_features_and_lengths,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    _dataset = _dataset.map(tf_serialize_example)

    return _dataset
