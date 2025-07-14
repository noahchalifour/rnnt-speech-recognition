import os
import tensorflow_datasets as tfds
import tensorflow as tf

from hparams import (
    HP_TOKEN_TYPE,
    HP_VOCAB_SIZE,
)
from . import vocabulary, preprocessing


def build_lookup_table(keys, values=None, default_value=-1):
    """Build a lookup table for mapping keys to values.

    Args:
        keys: A list of keys for the lookup table.
        values: Optional list of values corresponding to the keys.
            If None, values will be set to range(len(keys)).
        default_value: Value to return for keys not in the table.

    Returns:
        A TensorFlow StaticHashTable.
    """
    if values is None:
        values = tf.range(len(keys))

    kv_init = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values)

    return tf.lookup.StaticHashTable(kv_init, default_value=default_value)


def wordpiece_encode(text, encoder):
    """Encode text using a wordpiece encoder.

    Args:
        text: A tensor containing text to encode.
        encoder: A SubwordTextEncoder instance.

    Returns:
        A tensor of encoded token IDs.
    """
    return tf.constant(encoder.encode(text.numpy()), dtype=tf.int32)


def tf_wordpiece_encode(text, encoder):
    """TensorFlow wrapper for wordpiece_encode function.

    Args:
        text: A tensor containing text to encode.
        encoder: A SubwordTextEncoder instance.

    Returns:
        A tensor of encoded token IDs.
    """

    def _encode_fn(x):
        return wordpiece_encode(x, encoder)

    return tf.py_function(_encode_fn, inp=[text], Tout=tf.int32)


def wordpiece_decode(ids, encoder):
    """Decode token IDs using a wordpiece encoder.

    Args:
        ids: A tensor containing token IDs to decode.
        encoder: A SubwordTextEncoder instance.

    Returns:
        A tensor containing the decoded text.
    """
    return tf.constant(encoder.decode(ids.numpy()))


def tf_wordpiece_decode(ids, encoder):
    """TensorFlow wrapper for wordpiece_decode function.

    Args:
        ids: A tensor containing token IDs to decode.
        encoder: A SubwordTextEncoder instance.

    Returns:
        A tensor containing the decoded text.
    """

    def _decode_fn(x):
        return wordpiece_decode(x, encoder)

    return tf.py_function(_decode_fn, inp=[ids], Tout=[tf.string])[0]


def tf_vocab_encode(text, vocab_table):
    """Encode text using a vocabulary lookup table.

    Args:
        text: A tensor containing text to encode.
        vocab_table: A TensorFlow lookup table mapping characters to token IDs.

    Returns:
        A tensor of encoded token IDs.
    """
    tokens = tf.strings.bytes_split(text)

    return vocab_table.lookup(tokens)


def get_encoder(encoder_dir, hparams, texts_generator=None):
    """Get encoder and decoder functions based on the tokenization type.

    Args:
        encoder_dir: Directory to store or load the encoder.
        hparams: Hyperparameters dictionary containing tokenization settings.
        texts_generator: Optional generator yielding text samples for building vocabulary.

    Returns:
        A tuple containing:
            - encoder_fn: Function to encode text to token IDs.
            - decoder_fn: Function to decode token IDs to text (None for character tokenization).
            - vocab_size: Size of the vocabulary.
    """

    def preprocessed_gen():
        """Generator that normalizes text from the texts_generator."""
        if texts_generator is None:
            return
        for x in texts_generator:
            yield preprocessing.normalize_text(x)

    if hparams[HP_TOKEN_TYPE.name] == "character":
        vocab = vocabulary.init_vocab()
        vocab_table = build_lookup_table(vocab, default_value=0)

        vocab_size = len(vocab)

        def character_encoder_fn(text):
            """Encode text using character-based vocabulary."""
            return tf_vocab_encode(text, vocab_table)

        # For character encoding, we don't have a decoder function
        def character_decoder_fn(ids):
            """Decode function for character-based encoding (not implemented)."""
            return None

        # Assign the character-specific functions
        encoder_fn = character_encoder_fn
        decoder_fn = character_decoder_fn
    elif hparams[HP_TOKEN_TYPE.name] == "word-piece":
        encoder_filename = "encoder"
        encoder_filepath = os.path.join(encoder_dir, encoder_filename)

        if os.path.exists("{}.subwords".format(encoder_filepath)):
            encoder = tfds.core.features.text.SubwordTextEncoder.load_from_file(
                encoder_filepath
            )
        else:
            encoder = tfds.core.features.text.SubwordTextEncoder.build_from_corpus(
                corpus_generator=preprocessed_gen(),
                target_vocab_size=hparams[HP_VOCAB_SIZE.name],
            )
            os.makedirs(encoder_dir, exist_ok=True)
            encoder.save_to_file(encoder_filepath)

        vocab_size = encoder.vocab_size

        def encoder_fn(text):
            """Encode text using wordpiece tokenization."""
            return tf_wordpiece_encode(text, encoder)

        def decoder_fn(ids):
            """Decode token IDs using wordpiece tokenization."""
            return tf_wordpiece_decode(ids, encoder)
    else:
        raise Exception

    return encoder_fn, decoder_fn, vocab_size
