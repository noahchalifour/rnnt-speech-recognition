def init_vocab():

    alphabet = "abcdefghijklmnopqrstuvwxyz'"
    alphabet_c = ['', ' ', '<s>', '</s>'] + [c for c in alphabet]

    return alphabet_c


def load_vocab(filepath):

    vocab = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().strip('\n')
            if line == '<blank>':
                line = ''
            elif line == '<space>':
                line = ' '
            vocab.append(line)

    return vocab


def save_vocab(vocab, filepath):

    with open(filepath, 'w') as f:
        for c in vocab:
            if c == '':
                c = '<blank>'
            elif c == ' ':
                c = '<space>'
            f.write('{}\n'.format(c))