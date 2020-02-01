from argparse import ArgumentParser
from tqdm import tqdm
from pydub import AudioSegment
import os


def mp3_to_wav(filepath):

    audio_segment = AudioSegment.from_mp3(filepath)
    audio_segment.export('{}.wav'.format(filepath[:-4]), format='wav')
    os.remove(filepath)


def main(args):

    print('Converting all Common Voice MP3s to WAV...')

    clips_dir = os.path.join(args.data_dir, 'clips')
    all_clips = os.listdir(clips_dir)

    for clip in tqdm(all_clips):

        if clip[-4:] != '.mp3':
            continue

        clip_fp = os.path.join(clips_dir, clip)
        mp3_to_wav(clip_fp)

    print('Done.')


def parse_args():

    ap = ArgumentParser()

    ap.add_argument('--data_dir', required=True, type=str,
        help='Path to common voice data directory.')
    
    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args)