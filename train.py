from absl import app, flags, logging
import os
import tensorflow as tf

from model import transducer
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dataset', './data/train', 'Training dataset.')
flags.DEFINE_string('val_dataset', './data/dev', 'Validation dataset.')

flags.DEFINE_integer('feat_size', 80, 'Features size.')
flags.DEFINE_integer('encoder_layers', 1, 'Number of encoder layers.')
flags.DEFINE_integer('encoder_size', 100, 'Units per encoder layer.')
flags.DEFINE_integer('pred_net_layers', 1, 'Number of prediction network layers.')
flags.DEFINE_integer('pred_net_size', 100, 'Units per prediction network layer.')
flags.DEFINE_integer('joint_net_size', 100, 'Joint network units.')
flags.DEFINE_integer('softmax_size', 30, 'Units in softmax layer.')

flags.DEFINE_integer('batch_size', 64, 'Batch size.')

def _parse_example(example_proto):

    context_features = {
        'seq_len': tf.io.FixedLenFeature([], tf.int64),
        'labels': tf.io.VarLenFeature(tf.int64),
        'labels_len': tf.io.FixedLenFeature([], tf.int64)
    }

    sequence_features = {
        'feats': tf.io.FixedLenSequenceFeature([80], tf.float32)
    }

    return tf.io.parse_single_sequence_example(
        example_proto, context_features, sequence_features)


def load_dataset(path):

    dataset = tf.data.Dataset.list_files(os.path.join(path, '*.tfrecords'), shuffle=False)
    
    dataset = dataset.interleave(tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(_parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.padded_batch(batch_size=FLAGS.batch_size,
    #     padded_shapes=[-1])
    dataset = dataset.batch(batch_size=FLAGS.batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def main(_):

    vocab = utils.init_vocab()

    train_dataset = load_dataset(FLAGS.train_dataset)
    val_dataset = load_dataset(FLAGS.val_dataset)

    logging.info('Initializing model from scratch.')

    model = transducer(vocab_size=len(vocab),
                       encoder_layers=FLAGS.encoder_layers,
                       encoder_size=FLAGS.encoder_size,
                       pred_net_layers=FLAGS.pred_net_layers,
                       pred_net_size=FLAGS.pred_net_size,
                       joint_net_size=FLAGS.joint_net_size,
                       softmax_size=FLAGS.softmax_size)

    logging.info('Begin training.')


if __name__ == '__main__':

    app.run(main)