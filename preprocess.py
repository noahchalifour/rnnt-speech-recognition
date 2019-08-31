from absl import app, logging, flags
from pydub import AudioSegment
import os
import sys
import time
import numpy as np
import tensorflow as tf
import python_speech_features as psf

FLAGS = flags.FLAGS

flags.DEFINE_enum('dataset', None, ['common-voice'], 'Dataset to preprocess.')
flags.DEFINE_string('data_dir', None, 'Dataset path.')
flags.DEFINE_string('out_dir', None, 'Output path.')
flags.DEFINE_boolean('sortagrad', True, 'Use sortagrad')

# Required flags
flags.mark_flag_as_required('dataset')
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('out_dir')

# Special labels
BLANK_LABEL = '<blank>'
SOS_LABEL = '<sos>'

def init_vocab():

    """Generate a TF Example from features and labels

    Args:
        feats: Log-mel filter banks
        seq_len: Length of feature sequence
        labels: Target labels

    Returns:
        Serialized TF Example
    """

    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    specials = [BLANK_LABEL, SOS_LABEL]

    return {c: i for c, i in zip(specials + [c for c in alphabet],
                                 range(len(alphabet) + len(specials)))}
    

def clean_transcription(text):

    """Normalize the text for training

    Args:
        text: String text

    Returns:
        Normalized text
    """

    return text.lower()


def text_to_ids(text):

    """Convert text to list of char ids

    Args:
        text: String text

    Returns:
        List of char ids
    """

    return np.array([VOCAB[SOS_LABEL]] + [VOCAB[c] for c in text
                                          if c in VOCAB])


def audio_segment_to_array(audio_segment):

    """Convert pydub.AudioSegment to Numpy array

    Args:
        audio_segment: pydub.AudioSegment

    Returns:
        Numpy array with audio data
    """

    samples = audio_segment.get_array_of_samples()
    arr = np.array(samples)

    return arr


def compute_filter_banks(audio_arr):

    """Compute Log-Mel filter banks from audio

    Args:
        audio_arr: Numpy audio array

    Returns:
        Log-Mel filter banks
    """

    return psf.logfbank(audio_arr, winlen=0.025, winstep=0.01,
                        nfilt=80)


def make_example(feats, seq_len, labels, labels_len):

    """Generate a TF Example from features and labels

    Args:
        feats: Log-mel filter banks
        seq_len: Length of feature sequence
        labels: Target labels

    Returns:
        Serialized TF Example
    """

    feats_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame))
                  for frame in feats]
    feat_dict = {'feats': tf.train.FeatureList(feature=feats_list)}
    sequence_feats = tf.train.FeatureLists(feature_list=feat_dict)

    seq_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
    label_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
    label_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[labels_len]))

    context_feats = tf.train.Features(feature={'seq_len': seq_len_feat,
                                               'labels': label_feat,
                                               'labels_len': label_len_feat})

    ex = tf.train.SequenceExample(context=context_feats,
                                  feature_lists=sequence_feats)

    return ex.SerializeToString()
    

def preprocess_common_voice(data_dir, out_dir, sortagrad=True):

    """Preprocess Common Voice dataset

    Args:
        data_dir: Data directory path
        out_dir: Output directory path
    """

    splits = ['train', 'dev', 'test']

    for data_split in splits:

        total_sample_count = 0
        
        record_sample_count = 0
        record_size = 0
        record_count = 0
        record_writer = None
        
        with open(os.path.join(data_dir, '{}.tsv'.format(data_split)), 'r') as f:

            f.readline()
            data = []
            
            for line in f:
                
                # Parse file line
                tab_sep = line.split('\t')
                audio_filepath = os.path.join(data_dir, 'clips', tab_sep[1])
                transcription = tab_sep[2]

                data.append((audio_filepath, transcription))

            if 'train' in data_split and sortagrad:
                # Perform Sortagrad on training data
                data.sort(key=lambda x: len(x[1]))
            
            for audio_filepath, transcription in data:

                # Initialize a new TF Record Writer
                if record_writer is None:
                    output_filename = 'common_voice_{}_{:06d}.tfrecords'.format(
                        data_split, record_count)
                    output_dir = os.path.join(out_dir, data_split)
                    output_path = os.path.join(output_dir, output_filename)
                    os.makedirs(output_dir, exist_ok=True)
                    record_writer = tf.io.TFRecordWriter(output_path)
                    record_count += 1

                # Preprocess audio
                audio_segment = AudioSegment.from_mp3(audio_filepath)
                audio_arr = audio_segment_to_array(audio_segment)
                filter_banks = compute_filter_banks(audio_arr)
                fb_length = filter_banks.shape[0]

                # Preprocess transcription
                transcription_c = clean_transcription(transcription)
                t_ids = text_to_ids(transcription_c)
                t_len = t_ids.shape[0]

                # Create TF Example
                example = make_example(filter_banks, fb_length, t_ids, t_len)
                record_writer.write(example)
                
                total_sample_count += 1
                record_sample_count += 1
                record_size += sys.getsizeof(example) / 1e+6

                # Write TF record file if over 150MB
                # From guide: https://www.tensorflow.org/beta/tutorials/load_data/tf_records
                if record_size >= 150:
                    logging.info('Wrote record file \'{}\' containing {} samples.'.format(
                        output_filename, record_sample_count))
                    record_sample_count = 0
                    record_size = 0
                    record_writer.close()
                    record_writer = None

            logging.info('Wrote record file \'{}\' containing {} samples.'.format(
                output_filename, record_sample_count))
            record_writer.close()

        logging.info('Completed \'{}\' data split, contains {} total samples split into {} record files.'.format(
            data_split, total_sample_count, record_count))
    

def main(_):

    # Initialize text to ids vocabulary
    global VOCAB
    VOCAB = init_vocab()

    start_time = time.time()

    # Preprocess functions map
    {'common-voice': preprocess_common_voice
    }[FLAGS.dataset](FLAGS.data_dir, FLAGS.out_dir, sortagrad=FLAGS.sortagrad)

    time_elapsed = time.time() - start_time

    logging.info('Preprocessing complete, took: {}s'.format(time_elapsed))


if __name__ == '__main__':

    app.run(main)
