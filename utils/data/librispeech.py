import os
import tensorflow as tf
import soundfile as sf


def load_audio(filepath):

    return sf.read(filepath)


def tf_load_audio(filepath):

    return tf.py_function(
        lambda x: load_audio(x.numpy()),
        inp=[filepath],
        Tout=[tf.float32, tf.int32])


def tf_file_exists(filepath):

    return tf.py_function(
        lambda x: os.path.exists(x.numpy()),
        inp=[filepath],
        Tout=tf.bool)


def tf_parse_line(line, data_dir, split_names):

    line_split = tf.strings.split(line, ' ')

    audio_fn = line_split[0]
    transcription = tf.py_function(
        lambda x: b' '.join(x.numpy()).decode('utf8'),
        inp=[line_split[1:]],
        Tout=tf.string)

    speaker_id, chapter_id, _ = tf.unstack(tf.strings.split(audio_fn, '-'), 3)

    all_fps = tf.map_fn(
        lambda split_name: tf.strings.join([data_dir, split_name, speaker_id, chapter_id, audio_fn], '/') + '.flac',
        tf.constant(split_names))

    audio_filepath_idx = tf.where(
        tf.map_fn(tf_file_exists, all_fps, dtype=tf.bool))[0][0]
    audio_filepath = all_fps[audio_filepath_idx]

    audio, sr = tf_load_audio(audio_filepath)

    return audio, sr, transcription


def get_transcript_files(base_path, split_names):

    transcript_files = []

    for split_name in split_names:
        for speaker_id in os.listdir(f'{base_path}/{split_name}'):
            if speaker_id == '.DS_Store': continue
            for chapter_id in os.listdir(f'{base_path}/{split_name}/{speaker_id}'):
                if chapter_id == '.DS_Store': continue
                transcript_files.append(f'{base_path}/{split_name}/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}.trans.txt')

    return transcript_files


def load_dataset(base_path, split_names):

    transcript_filepaths = get_transcript_files(base_path, split_names)

    dataset = tf.data.TextLineDataset(transcript_filepaths)
    dataset = dataset.map(lambda line: tf_parse_line(line, base_path, split_names),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def texts_generator(base_path, split_names):

    transcript_filepaths = get_transcript_files(base_path, split_names)
    for fp in transcript_filepaths:
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip('\n')
                transcription = ' '.join(line.split(' ')[1:])
                yield transcription