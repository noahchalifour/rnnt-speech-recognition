"""
Vocabulary utilities for speech recognition.

This module provides functions for handling vocabularies in speech recognition tasks,
including initialization, loading from files, and saving to files. It handles special
tokens like blank, space, start-of-sequence, and end-of-sequence.
"""


def init_vocab():
    """
    Initialize a default English alphabet vocabulary.
    
    Creates a vocabulary containing the English alphabet (a-z), apostrophe ('),
    and special tokens: blank (''), space (' '), start-of-sequence ('<s>'),
    and end-of-sequence ('</s>').
    
    Returns:
        list: A list of characters and special tokens representing the initialized vocabulary.
        
    Example:
        >>> vocab = init_vocab()
        >>> print(len(vocab))  # 31 (26 letters + ' + 4 special tokens)
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz'"
    alphabet_c = ['', ' ', '<s>', '</s>'] + [c for c in alphabet]

    return alphabet_c


def load_vocab(filepath):
    """
    Load vocabulary from a file.
    
    Reads a vocabulary from a text file with each token on a separate line.
    Special tokens are handled as follows:
    - '<blank>' is converted to empty string ''
    - '<space>' is converted to space character ' '
    
    Args:
        filepath (str): Path to the vocabulary file to be loaded.
        
    Returns:
        list: A list of characters and special tokens representing the loaded vocabulary.
        
    Raises:
        IOError: If the file cannot be read from the specified location.
        
    Example:
        >>> vocab = load_vocab('path/to/vocab.txt')
    """
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
    """
    Save vocabulary to a file.
    
    Writes a vocabulary to a text file with each token on a separate line.
    Special tokens are handled as follows:
    - Empty string '' is converted to '<blank>'
    - Space character ' ' is converted to '<space>'
    
    Args:
        vocab (list): List of characters and special tokens representing the vocabulary.
        filepath (str): Path where the vocabulary file will be saved.
        
    Raises:
        IOError: If the file cannot be written to the specified location.
    """
    with open(filepath, 'w') as f:
        for c in vocab:
            if c == '':
                c = '<blank>'
            elif c == ' ':
                c = '<space>'
            f.write('{}\n'.format(c))
