from argparse import ArgumentParser
from tqdm import tqdm
from pydub import AudioSegment
import os


def mp3_to_wav(filepath):

    try:
        audio_segment = AudioSegment.from_mp3(filepath)
        audio_segment.export('{}.wav'.format(filepath[:-4]), format='wav')
    except Exception:
        pass

    os.remove(filepath)


def remove_missing(data_dir, fname):

    clips_dir = os.path.join(data_dir, 'clips')

    old_filepath = os.path.join(data_dir, '{}.tsv'.format(fname))
    new_filepath = os.path.join(data_dir, '{}-tmp.tsv'.format(fname))

    with open(old_filepath, 'r') as old_f:
        with open(new_filepath, 'w') as new_f:
            new_f.write(next(old_f))
            for line in old_f:
                audio_fn = line.split('\t')[1][:-4] + '.wav'
                if os.path.exists(os.path.join(clips_dir, audio_fn)):
                    new_f.write(line)

    os.remove(old_filepath)
    os.rename(new_filepath, old_filepath)


def main(args):

    print('Converting all Common Voice MP3s to WAV...')

    clips_dir = os.path.join(args.data_dir, 'clips')
    all_clips = os.listdir(clips_dir)

    for clip in tqdm(all_clips):

        if clip[-4:] != '.mp3':
            continue

        clip_fp = os.path.join(clips_dir, clip)
        mp3_to_wav(clip_fp)

    print('Removing missing files...')

    tsv_files = ['dev', 'invalidated', 'other', 'test', 'train', 'validated']

    for _file in tsv_files:
        remove_missing(args.data_dir, _file)

    print('Done.')


def parse_args():

    ap = ArgumentParser()

    ap.add_argument('--data_dir', required=True, type=str,
        help='Path to common voice data directory.')
    
    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args)