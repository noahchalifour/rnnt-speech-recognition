from pydub import AudioSegment
import os
import tensorflow as tf

from . import common as cm_data_utils

def mp3_to_wav(filepath):

    audio_segment = AudioSegment.from_mp3(filepath)
    audio_segment.export('{}.wav'.format(filepath[:-4]), format='wav')
    os.remove(filepath)


def tf_file_exists(filepath):

    return tf.py_function(lambda x: os.path.exists(x.numpy().decode('utf8')), inp=[filepath], Tout=tf.bool)


def tf_parse_line(line, datapath):

    line_sections = tf.strings.split(line, '\t')

    audio_fn = line_sections[1]
    transcription = line_sections[2]
    audio_filepath = tf.strings.join([datapath, 'clips', audio_fn], '/')

    if tf.strings.regex_full_match(audio_fn, '(.*)\\.mp3'):
        wav_filepath = tf.strings.substr(audio_filepath, 0, tf.strings.length(audio_filepath) - 4) + '.wav'
        if tf.logical_not(tf_file_exists(wav_filepath)):
            tf.py_function(lambda x: mp3_to_wav(x.numpy().decode('utf8')),
                inp=[audio_filepath], Tout=[])
        audio_filepath = wav_filepath
        
    audio, sr = cm_data_utils.tf_load_audio(audio_filepath)

    return audio, sr, transcription


def _create_dataset(path, name, max_data=None):

    dataset = tf.data.TextLineDataset(
        [os.path.join(path, '{}.tsv'.format(name))])

    dataset = dataset.skip(1)
    dataset = dataset.map(lambda line: tf_parse_line(line, path),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if max_data is not None:
        dataset = dataset.take(max_data)

    return dataset


def create_datasets(path, max_data=None):

    train_dataset = _create_dataset(path, 'train', 
        max_data=max_data)
    dev_dataset = _create_dataset(path, 'dev', 
        max_data=max_data)

    return train_dataset, dev_dataset