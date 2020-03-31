from argparse import ArgumentParser
import os
import tensorflow as tf

from utils import model as model_utils


def main(args):

    hparams = model_utils.load_hparams(args.model_dir)
    model, _ = model_utils.load_model(args.model_dir, hparams,
        stateful=True)

    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    # converter.experimental_new_quantizer = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_quant_model = converter.convert()

    tflite_dir = os.path.join(args.model_dir, 'tflite')
    os.makedirs(tflite_dir, exist_ok=True)

    with open(os.path.join(tflite_dir, 'model.tflite'), 'wb') as f:
        f.write(tflite_quant_model)

def parse_args():

    ap = ArgumentParser()

    ap.add_argument('-m', '--model_dir', type=str, default='./model',
        help='Directory of model.')

    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args)