from pydub import AudioSegment
import os
import tensorflow as tf

from . import common as cm_data_utils

def load_audio(filename, datapath):

    audio_segment = AudioSegment.from_mp3(
        os.path.join(datapath, 'clips', filename))

    samples = audio_segment.get_array_of_samples()
    
    return tf.constant(samples)


def tf_parse_line(line, datapath):

    line_sections = tf.strings.split(line, '\t')

    audio_fn = line_sections[1]
    transcription = line_sections[2]

    audio_arr = tf.py_function(lambda x: load_audio(x.numpy().decode('utf8'), datapath),
        inp=[audio_fn], Tout=tf.int32)

    return audio_arr, transcription


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