# RNN-Transducer Speech Recognition

End-to-end speech recognition using RNN-Transducer in Tensorflow 2.0

## Overview

This speech recognition model is based off Google's [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/pdf/1811.06621.pdf) research paper and is implemented in Python 3 using Tensorflow 2.0

## Usage

The main script of the repository is the `run_rnnt.py` script. Everything is run through there, you are just going to be specifying a variety of parameters.

<i>Here is a list of all parameters:</i>

```
python run_rnnt.py --help

       USAGE: run_rnnt.py [flags]
flags:

run_rnnt.py:

  --batch_size: Batch size.
    (default: '64')
    (an integer)

  --checkpoint: Path of checkpoint to load (default to latest in 'model_dir')

  --dataset_name: <common-voice>: Dataset to use.

  --dataset_path: Dataset path.
  
  --encoder_layers: Number of encoder layers.
    (default: '8')
    (an integer)

  --encoder_size: Units per encoder layer.
    (default: '2048')
    (an integer)

  --epochs: Number of training epochs.
    (default: '20')
    (an integer)

  --eval_size: Eval size.
    (default: '1000')
    (an integer)

  --input: Input file.

  --joint_net_size: Joint network units.
    (default: '640')
    (an integer)

  --keep_top: Maximum checkpoints to keep.
    (default: '5')
    (an integer)

  --learning_rate: Training learning rate.
    (default: '0.0001')
    (a number)

  --max_data: Max size of data.
    (an integer)

  --mode: <train|eval|transcribe-file>: Mode to run in.

  --model_dir: Model output directory.
    (default: './model')

  --pred_net_layers: Number of prediction network layers.
    (default: '2')
    (an integer)

  --pred_net_size: Units per prediction network layer.
    (default: '2048')
    (an integer)

  --shuffle_buffer_size: Shuffle buffer size.
    (an integer)

  --softmax_size: Units in softmax layer.
    (default: '4096')
    (an integer)

  --steps_per_checkpoint: Number of steps between each checkpoint.
    (default: '1000')
    (an integer)
    
  --steps_per_log: Number of steps between each log written.
    (default: '100')
    (an integer)

  --tb_log_dir: Tensorboard log directory.
    (default: './logs')

  --tpu: GCP TPU to use.
```

## Getting Started

To setup your environment, run the following commands:

```
git clone --recurse https://github.com/noahchalifour/rnnt-speech-recognition.git
cd rnnt-speech-recognition
pip install tensorflow==2.1.0rc2 # or tensorflow-gpu==2.1.0rc2 for GPU support
pip install -r requirements.txt
```

Once your environment is all set you are ready to start training your own models.

## Supported Datasets

Currently we only support the [Common Voice](https://voice.mozilla.org/en/datasets) dataset. We plan on adding support for other datasets in the future.

## Training a Model

### Training on Host

To train a simple model, run the following command:

```
python run_rnnt.py \
    --mode train \
    --dataset_name common-voice \
    --dataset_path <path to your dataset>
```

### Training in Docker Container

[View Image](https://hub.docker.com/r/noahchalifour/rnnt-speech-recognition)

You can also train your model in a docker container based on the Tensorflow docker image. To do so, run the following commands:

> **_NOTE:_** Specify all your paramters in ALL CAPS as environment variables when training in a docker container.

```
docker run -d --name rnnt-speech-recognition \
    -e MODE=train \
    -e DATASET_NAME common-voice \
    -e DATASET_PATH <path to your dataset> \
    noahchalifour/rnnt-speech-recognition
```

## Evaluation

To run evaluation, use the following command:

```
python run_rnnt.py \
    --mode eval \
    --dataset_name common-voice \
    --dataset_path <path to your dataset>
```

## Inference

### Transcribing a WAV file

To transcribe a WAV file, run the following command:

```
python run_rnnt.py \
    --mode transcribe-file \
    --input <path to wav file>
```

### Real-time transcription

To run real-time transcription using your computer microphone, run the following command:

```
python run_rnnt.py \
    --mode realtime
```

## Author

Noah Chalifour, chalifournoah@gmail.com