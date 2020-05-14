import os
import tensorflow_datasets as tfds
import tensorflow as tf

from hparams import *
from . import vocabulary, preprocessing


def build_lookup_table(keys, values=None, default_value=-1):

    if values is None:
        values = tf.range(len(keys))

    kv_init = tf.lookup.KeyValueTensorInitializer(
        keys=keys, values=values)

    return tf.lookup.StaticHashTable(kv_init,
        default_value=default_value)


def wordpiece_encode(text, encoder):

    return tf.constant(encoder.encode(text.numpy()),
        dtype=tf.int32)


def tf_wordpiece_encode(text, encoder):

    return tf.py_function(lambda x: wordpiece_encode(x, encoder),
        inp=[text], Tout=tf.int32)


def wordpiece_decode(ids, encoder):

    return tf.constant(encoder.decode(ids.numpy()))


def tf_wordpiece_decode(ids, encoder):

    return tf.py_function(lambda x: wordpiece_decode(x, encoder),
        inp=[ids], Tout=[tf.string])[0]


def tf_vocab_encode(text, vocab_table):

    tokens = tf.strings.bytes_split(text)

    return vocab_table.lookup(tokens)


def get_encoder(encoder_dir,
                hparams,
                texts_generator=None):

    def preprocessed_gen():
        if texts_generator is None:
            return
        for x in texts_generator:
            yield preprocessing.normalize_text(x)

    if hparams[HP_TOKEN_TYPE.name] == 'character':

        vocab = vocabulary.init_vocab()
        vocab_table = build_lookup_table(vocab,
            default_value=0)

        vocab_size = len(vocab)

        encoder_fn = lambda text: tf_vocab_encode(text, vocab_table)
        decoder_fn = None

    elif hparams[HP_TOKEN_TYPE.name] == 'word-piece':

        encoder_filename = 'encoder'
        encoder_filepath = os.path.join(encoder_dir, encoder_filename)

        if os.path.exists('{}.subwords'.format(encoder_filepath)):
            encoder = tfds.features.text.SubwordTextEncoder.load_from_file(encoder_filepath)
        else:
            encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                corpus_generator=preprocessed_gen(),
                target_vocab_size=hparams[HP_VOCAB_SIZE.name])
            os.makedirs(encoder_dir, exist_ok=True)
            encoder.save_to_file(encoder_filepath)

        vocab_size = encoder.vocab_size

        encoder_fn = lambda text: tf_wordpiece_encode(text, encoder)
        decoder_fn = lambda ids: tf_wordpiece_decode(ids, encoder)

    return encoder_fn, decoder_fn, vocab_size
