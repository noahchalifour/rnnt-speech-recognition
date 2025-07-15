import os
import subprocess
import concurrent.futures
import tensorflow as tf
from absl import app, logging, flags
from tqdm import tqdm
from utils import preprocessing, encoding
from utils.data import common_voice
from hparams import *

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_dir', None,
    'Directory to read Common Voice data from.')
flags.DEFINE_string(
    'output_dir', './data',
    'Directory to save preprocessed data.')
flags.DEFINE_integer(
    'max_length', 0,
    'Max audio length in seconds.')
flags.DEFINE_integer(
    'num_workers', -1,
    'Number of parallel workers for audio conversion.')
flags.DEFINE_boolean(
    'replace_old', False,
    'Replace old TSV files with updated ones.')


def convert_audio_files(data_dir, num_workers=-1):
    """Convert MP3 files to WAV format using multiple threads."""
    clips_dir = os.path.join(data_dir, 'clips')
    if not os.path.exists(clips_dir):
        logging.error(f"Clips directory not found: {clips_dir}")
        return

    mp3_files = [f for f in os.listdir(clips_dir) if f.endswith('.mp3')]
    if not mp3_files:
        logging.warning(f"No MP3 files found in {clips_dir}")
        return

    def convert_file(mp3_file):
        wav_file = mp3_file[:-4] + '.wav'
        mp3_path = os.path.join(clips_dir, mp3_file)
        wav_path = os.path.join(clips_dir, wav_file)

        logging.info(f'Converting file {mp3_path}')
        
        try:
            # Run ffmpeg with suppressed output
            subprocess.run([
                'ffmpeg',
                '-i', mp3_path,
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-loglevel', 'quiet',
                wav_path
            ], check=True, capture_output=True)
            os.remove(mp3_path)
            logging.info('succeeded')
            return True
        except subprocess.CalledProcessError:
            logging.error('failed')
            return False

    max_workers = None if num_workers == -1 else num_workers

    logging.info(f"Starting audio conversion with {num_workers} workers...")
    
    # Create a progress bar
    pbar = tqdm(total=len(mp3_files), desc='Converting audio files', unit='file')

    def process_future(future):
        try:
            result = future.result()
            logging.info(f'Got result: {result}')
            results.append(result)
            pbar.update(1)
        except Exception as e:
            logging.error(f"Error processing: {str(e)}")
            results.append(False)
            pbar.update(1)

    # Create results list
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(convert_file, mp3_file) for mp3_file in mp3_files]
        
        # Process futures as they complete
        for future in concurrent.futures.as_completed(futures):
            process_future(future)
    
    pbar.close()
    
    success_count = sum(results)
    logging.info(f"Successfully converted {success_count}/{len(mp3_files)} files")
    if len(mp3_files) - success_count > 0:
        logging.warning(f"Failed to convert {len(mp3_files) - success_count} files")


def remove_missing_samples(data_dir, replace_old=False):
    """Remove entries from TSV files that don't have corresponding audio files."""
    clips_dir = os.path.join(data_dir, 'clips')
    
    tsv_files = ['dev', 'invalidated', 'other', 'test', 'train', 'validated']
    
    for tsv in tsv_files:
        old_filepath = os.path.join(data_dir, f'{tsv}.tsv')
        new_filepath = os.path.join(data_dir, f'{tsv}-tmp.tsv')
        
        if not os.path.exists(old_filepath):
            logging.warning(f"TSV file not found: {old_filepath}")
            continue
        
        with open(old_filepath, 'r') as old_f, open(new_filepath, 'w') as new_f:
            header = next(old_f)
            new_f.write(header)
            
            for line in old_f:
                audio_fn = line.split('\t')[1].strip()[:-4] + '.wav'
                audio_path = os.path.join(clips_dir, audio_fn)
                if os.path.exists(audio_path):
                    new_f.write(line)
        
        if replace_old:
            os.remove(old_filepath)
            os.rename(new_filepath, old_filepath)
        else:
            logging.info(f"Created new file: {new_filepath}")


def write_dataset(dataset, name):
    """Write dataset to TFRecord file."""
    filepath = os.path.join(FLAGS.output_dir, f'{name}.tfrecord')
    writer = tf.data.experimental.TFRecordWriter(filepath)
    writer.write(dataset)
    logging.info(f'Wrote {name} dataset to {filepath}')


def main(_):
    """Main preprocessing pipeline."""
    # Step 1: Convert audio files
    logging.info("Converting MP3 files to WAV format...")
    convert_audio_files(FLAGS.data_dir, FLAGS.num_workers)
    
    # Step 2: Remove missing samples from TSV files
    logging.info("Removing missing samples from TSV files...")
    remove_missing_samples(FLAGS.data_dir, FLAGS.replace_old)
    
    # Step 3: Preprocess data with TensorFlow
    logging.info("Starting TensorFlow preprocessing...")
    
    hparams = {
        HP_TOKEN_TYPE: HP_TOKEN_TYPE.domain.values[1],
        HP_VOCAB_SIZE: HP_VOCAB_SIZE.domain.values[0],
        HP_MEL_BINS: HP_MEL_BINS.domain.values[0],
        HP_FRAME_LENGTH: HP_FRAME_LENGTH.domain.values[0],
        HP_FRAME_STEP: HP_FRAME_STEP.domain.values[0],
        HP_HERTZ_LOW: HP_HERTZ_LOW.domain.values[0],
        HP_HERTZ_HIGH: HP_HERTZ_HIGH.domain.values[0],
        HP_DOWNSAMPLE_FACTOR: HP_DOWNSAMPLE_FACTOR.domain.values[0]
    }

    _hparams = {k.name: v for k, v in hparams.items()}

    texts_gen = common_voice.texts_generator(FLAGS.data_dir)
    encoder_fn, decoder_fn, vocab_size = encoding.get_encoder(
        encoder_dir=FLAGS.output_dir,
        hparams=_hparams,
        texts_generator=texts_gen)
    _hparams[HP_VOCAB_SIZE.name] = vocab_size

    train_dataset = common_voice.load_dataset(FLAGS.data_dir, 'train')
    dev_dataset = common_voice.load_dataset(FLAGS.data_dir, 'dev')
    test_dataset = common_voice.load_dataset(FLAGS.data_dir, 'test')

    train_dataset = preprocessing.preprocess_dataset(
        train_dataset,
        encoder_fn=encoder_fn,
        hparams=_hparams,
        max_length=FLAGS.max_length,
        save_plots=True)
    write_dataset(train_dataset, 'train')

    dev_dataset = preprocessing.preprocess_dataset(
        dev_dataset,
        encoder_fn=encoder_fn,
        hparams=_hparams,
        max_length=FLAGS.max_length)
    write_dataset(dev_dataset, 'dev')

    test_dataset = preprocessing.preprocess_dataset(
        test_dataset,
        encoder_fn=encoder_fn,
        hparams=_hparams,
        max_length=FLAGS.max_length)
    write_dataset(test_dataset, 'test')

    logging.info("Preprocessing complete!")


if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    app.run(main)
