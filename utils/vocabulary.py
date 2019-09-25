def init_vocab():

    alphabet = " abcdefghijklmnopqrstuvwxyz'"
    alphabet_c = ['', '<s>', '</s>'] + [c for c in alphabet]

    return {alphabet_c[i]: i for i in range(len(alphabet_c))}