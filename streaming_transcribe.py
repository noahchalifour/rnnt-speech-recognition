from argparse import ArgumentParser
import os
import time
import pyaudio

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from utils import preprocessing, encoding, decoding
from utils import model as model_utils
from model import build_keras_model
from hparams import *


SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CHUNK_SIZE = 1024

LAST_OUTPUT = ''


def main(args):

    model_dir = os.path.dirname(os.path.realpath(args.checkpoint))

    hparams = model_utils.load_hparams(model_dir)

    _, tok_to_text, vocab_size = encoding.get_encoder(
        encoder_dir=model_dir,
        hparams=hparams)
    hparams[HP_VOCAB_SIZE.name] = vocab_size

    model = build_keras_model(hparams, stateful=True)
    model.load_weights(args.checkpoint)

    decoder_fn = decoding.greedy_decode_fn(model, hparams)

    p = pyaudio.PyAudio()

    def listen_callback(in_data, frame_count, time_info, status):
        global LAST_OUTPUT

        audio = tf.io.decode_raw(in_data, out_type=tf.float32)

        log_melspec = preprocessing.preprocess_audio(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            hparams=hparams)
        log_melspec = tf.expand_dims(log_melspec, axis=0)

        decoded = decoder_fn(log_melspec)[0]

        transcription = LAST_OUTPUT + tok_to_text(decoded)\
            .numpy().decode('utf8')

        if transcription != LAST_OUTPUT:
            LAST_OUTPUT = transcription
            print(transcription)

        return in_data, pyaudio.paContinue

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=NUM_CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=listen_callback)

    print('Listening...')

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()

    p.terminate()


def parse_args():

    ap = ArgumentParser()

    ap.add_argument('--checkpoint', type=str, required=True,
        help='Checkpoint to load.')

    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args)
