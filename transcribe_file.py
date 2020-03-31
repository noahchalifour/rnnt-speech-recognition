from argparse import ArgumentParser
import os
import tensorflow as tf

from utils import preprocessing, encoding, decoding
from utils import model as model_utils
from model import build_keras_model
from hparams import *


def main(args):

    model_dir = os.path.dirname(os.path.realpath(args.checkpoint))

    hparams = model_utils.load_hparams(model_dir)

    encode_fn, tok_to_text, vocab_size = encoding.load_encoder(model_dir, 
        hparams=hparams)
    hparams[HP_VOCAB_SIZE.name] = vocab_size

    start_token = encode_fn('')[0]

    model = build_keras_model(hparams)
    model.load_weights(args.checkpoint)
    
    audio, sr = preprocessing.tf_load_audio(args.input)

    mel_specs = preprocessing.compute_mel_spectrograms(
        audio_arr=audio,
        sample_rate=sr,
        n_mel_bins=hparams[HP_MEL_BINS.name],
        frame_length=hparams[HP_FRAME_LENGTH.name],
        frame_step=hparams[HP_FRAME_STEP.name],
        hertz_low=hparams[HP_HERTZ_LOW.name],
        hertz_high=hparams[HP_HERTZ_HIGH.name])

    mel_specs = tf.expand_dims(mel_specs, axis=0)

    decoder_fn = decoding.greedy_decode_fn(model, 
        start_token=start_token)

    decoded = decoder_fn(mel_specs)[0]
    transcription = tok_to_text(decoded)

    print('Transcription:', transcription.numpy().decode('utf8'))


def parse_args():

    ap = ArgumentParser()

    ap.add_argument('--checkpoint', type=str, required=True, 
        help='Checkpoint to load.')
    ap.add_argument('-i', '--input', type=str, required=True,
        help='Wav file to transcribe.')

    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args)