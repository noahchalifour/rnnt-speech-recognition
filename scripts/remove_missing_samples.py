from argparse import ArgumentParser
import os


def remove_missing(data_dir, fname, replace_old=True):

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

    if replace_old:
        os.remove(old_filepath)
        os.rename(new_filepath, old_filepath)


def main(args):

    tsv_files = ['dev', 'invalidated', 'other', 
                 'test', 'train', 'validated']

    for _file in tsv_files:
        remove_missing(args.data_dir, _file,
            replace_old=args.replace_old)

    print('Done.')


def parse_args():

    ap = ArgumentParser()

    ap.add_argument('--data_dir', required=True, type=str,
        help='Path to common voice data directory.')
    ap.add_argument('--replace_old', type=bool, default=False,
        help='Replace old tsv files with updated ones.')
    
    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args)