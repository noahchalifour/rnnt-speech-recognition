def init_vocab():

    alphabet = "abcdefghijklmnopqrstuvwxyz'"
    alphabet_c = ['<blank>', '<space>', '<s>', '</s>'] + [c for c in alphabet]

    return {alphabet_c[i]: i for i in range(len(alphabet_c))}


def load_vocab(filepath):

    vocab = {}

    with open(filepath, 'r') as f:
        for line in f:
            _line = line.strip().strip('\n')
            line_sep = _line.split(' ')
            vocab[line_sep[0]] = int(line_sep[1])

    return vocab


def save_vocab(vocab, filepath):

    with open(filepath, 'w') as f:
        for k, v in vocab.items():
            f.write('{} {}\n'.format(k, v))