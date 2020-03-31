from argparse import ArgumentParser
from scipy.io.wavfile import read as read_wav
import glob
import os


def main(args):

    max_length = 0
    min_length = 0
    total_length = 0
    count = 0

    with open(os.path.join(args.data_dir, args.split + '.tsv'), 'r') as f:
        next(f)
        for line in f:

            line_split = line.split('\t')
            audio_fn = line_split[1]

            filepath = os.path.join(args.data_dir, 'clips', audio_fn[:-4] + '.wav')

            sr, data = read_wav(filepath)

            length = len(data) / sr

            if length > max_length:
                max_length = length
            if length < min_length or min_length == 0:
                min_length = length

            total_length += length
            count += 1

    avg_length = total_length / count

    print('Total: {:.4f} s'.format(total_length))
    print('Min length: {:.4f} s'.format(min_length))
    print('Max length: {:.4f} s'.format(max_length))
    print('Average length: {:.4f} s'.format(avg_length))


def parse_args():

    ap = ArgumentParser()

    ap.add_argument('-d', '--data_dir', required=True, type=str,
        help='Directory of common voice dataset.')
    ap.add_argument('-s', '--split', type=str, default='train',
        help='Split to get statistics for.')

    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args)