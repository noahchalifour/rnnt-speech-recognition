import os
import tensorflow_datasets as tfds
import tensorflow as tf

from hparams import *
from . import vocabulary


def build_lookup_table(keys, values=None, default_value=-1):

    if values is None:
        values = tf.range(len(keys))

    kv_init = tf.lookup.KeyValueTensorInitializer(
        keys=keys, values=values)

    return tf.lookup.StaticHashTable(kv_init,
        default_value=default_value)


def wordpiece_encode(text, encoder):

    return tf.constant(
        encoder.encode(b'<s> ' + text.numpy() + b' </s>'))


def tf_wordpiece_encode(text, encoder):

    return tf.py_function(lambda x: wordpiece_encode(x, encoder), 
        inp=[text], Tout=[tf.int32])[0]


def vocab_encode(text, vocab_table):

    tokens = tf.strings.bytes_split(text)
    tokens = tf.concat([['<s>'], tokens, ['</s>']], axis=0)

    return vocab_table.lookup(tokens)


def build_encoder(texts_generator, model_dir, hparams):

    if hparams[HP_TOKEN_TYPE.name] == 'character':

        vocab = vocabulary.init_vocab()
        vocab_table = build_lookup_table(vocab, 
            default_value=0)

        vocab_size = len(vocab)

        encoder_fn = lambda text: vocab_encode(text, vocab_table)

    elif hparams[HP_TOKEN_TYPE.name] == 'word-piece':

        encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            corpus_generator=texts_generator, target_vocab_size=hparams[HP_VOCAB_SIZE.name],
            reserved_tokens=['<pad>', '<s>', '</s>'])

        vocab_size = encoder.vocab_size

        os.makedirs(model_dir, exist_ok=True)
        encoder.save_to_file(os.path.join(model_dir, 'encoder'))

        encoder_fn = lambda text: tf_wordpiece_encode(text, encoder)

    return encoder_fn, vocab_size


def load_encoder(model_dir, hparams):

    if hparams[HP_TOKEN_TYPE.name] == 'character':

        vocab = vocabulary.init_vocab()
        vocab_table = build_lookup_table(vocab, 
            default_value=0)

        vocab_size = len(vocab)

        return vocab_table.lookup, vocab_size

    elif hparams[HP_TOKEN_TYPE.name] == 'word-piece':

        encoder_fp = os.path.join(model_dir, 'encoder')
        encoder = tfds.features.text.SubwordTextEncoder.load_from_file(encoder_fp)

        vocab_size = encoder.vocab_size

        return encoder.encode, vocab_size
