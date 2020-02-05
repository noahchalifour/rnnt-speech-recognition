from argparse import ArgumentParser
from tqdm import tqdm
from pydub import AudioSegment
import functools
import multiprocessing
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


def mp3_converter_job(mp3_filenames, counter):

    for filename in mp3_filenames:

        if filename[-4:] == '.mp3':
            mp3_to_wav(filename)

        # with counter.get_lock():
        counter.value += 1


def show_progress_bar(val, total):

    prog = tqdm(total=total)
    while True:
        if not val.value:
            continue
        prog.n = val.value
        prog.update(0)
        if val.value >= total:
            break


def main(args):

    print('Converting all Common Voice MP3s to WAV...')

    convert_manager = multiprocessing.Manager()
    convert_counter = convert_manager.Value('i', 0,
        lock=convert_manager.Lock())

    clips_dir = os.path.join(args.data_dir, 'clips')

    all_clips = os.listdir(clips_dir)
    all_clips = [os.path.join(clips_dir, clip) for clip in all_clips]

    num_total = len(all_clips)

    num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cpus)

    job_size = num_total // num_cpus

    jobs = []
    for _ in range(num_cpus - 1):
        jobs.append(all_clips[:job_size])
        all_clips[job_size:]

    jobs.append(all_clips)
    all_clips = []

    mp3_partial = functools.partial(mp3_converter_job, counter=convert_counter)

    progress = multiprocessing.Process(target=show_progress_bar, args=(convert_counter, num_total))
    progress.start()

    pool.map_async(mp3_partial, jobs)

    pool.close()
    pool.join()

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