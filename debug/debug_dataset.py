from argparse import ArgumentParser
import os
import json
import sys
import tensorflow as tf

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILE_DIR, '..'))

from utils import preprocessing


def check_for_invalid_values(inp, labels):

    tf.debugging.check_numerics(inp['mel_specs'],
        message='mel_specs has invalid value.')

    return inp, labels


def check_empty(inp, labels):

    tf.debugging.assert_none_equal(
        tf.size(inp['mel_specs']), 0,
        message='mel_specs is empty tensor.')

    tf.debugging.assert_none_equal(
        tf.size(inp['pred_inp']), 0,
        message='pred_inp is empty tensor.')

    tf.debugging.assert_none_equal(
        tf.size(inp['spec_lengths']), 0,
        message='spec_lengths is empty tensor.')

    tf.debugging.assert_none_equal(
        tf.size(inp['label_lengths']), 0,
        message='label_lengths is empty tensor.')

    tf.debugging.assert_none_equal(
        tf.size(labels), 0,
        message='labels is empty tensor.')

    return inp, labels


def get_dataset(data_dir, 
                name, 
                batch_size,
                n_epochs):

    dataset = preprocessing.load_dataset(data_dir, name)
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=({
            'mel_specs': [-1, -1], 
            'pred_inp': [-1],
            'spec_lengths': [],
            'label_lengths': []
        }, [-1]))

    dataset = dataset.repeat(n_epochs)

    with open(os.path.join(data_dir, '{}-specs.json'.format(name)), 'r') as f:
        dataset_specs = json.load(f)

    return dataset, dataset_specs


def main(args):

    dataset, dataset_specs = get_dataset(
        args.data_dir, args.split,
        batch_size=1, n_epochs=1)

    dataset.map(check_for_invalid_values)
    dataset.map(check_empty)

    for _ in dataset:
        pass
    
    print('All checks passed.')


def parse_args():

    ap = ArgumentParser()

    ap.add_argument('-d', '--data_dir', type=str, required=True,
        help='Path to preprocessed dataset.')
    ap.add_argument('-s', '--split', type=str, default='train',
        help='Name of dataset split to inspect.')

    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args)