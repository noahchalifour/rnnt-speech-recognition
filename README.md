# RNN-Transducer Speech Recognition

End-to-end speech recognition using RNN-Transducer in Tensorflow 2.0

## Overview

This speech recognition model is based off Google's [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/pdf/1811.06621.pdf) research paper and is implemented in Python 3 using Tensorflow 2.0

> **_NOTE:_** If you are not training using docker you must run the following commands + setup the loss function (instructions for this can be found in `warp-transducer/tensorflow_binding`)

## Common Voice

### Preprocessing Data

Before you can train a model on the Common Voice dataset, you must first run the `preprocess_common_voice.py` script. Do so by running the following command:

```
python preprocess_common_voice.py --data_dir <path to dataset>
```

### Training a model

To setup your environment, run the following commands:

```
git clone --recurse https://github.com/noahchalifour/rnnt-speech-recognition.git
cd rnnt-speech-recognition
pip install tensorflow==2.1.0 # or tensorflow-gpu==2.1.0 for GPU support
pip install -r requirements.txt
```

Once your environment is all set you are ready to start training your own models.

#### Training on Host

To train a simple model, run the following command:

```
python run_common_voice.py \
    --mode train \
    --data_dir <path to common voice directory>
```

#### Training in Docker Container

[View Image](https://hub.docker.com/r/noahchalifour/rnnt-speech-recognition)

You can also train your model in a docker container based on the Tensorflow docker image. 

> **_NOTE:_** Specify all your paramters in ALL CAPS as environment variables when training in a docker container.

To run the model using a CPU only, run the following command:

```
docker run -d --name rnnt-speech-recognition \
    -v <path to local data>:/rnnt-speech-recognition/data \
    -v <path to save model locally>:/rnnt-speech-recognition/model \
    -e MODE=train \
    -e DATA_DIR=./data \
    -e MODEL_DIR=./model \
    noahchalifour/rnnt-speech-recognition
```

To run the model using a GPU you must run the following command with the added `--cap-add SYS_ADMIN`, and `--gpus <gpus>`:

```
docker run -d --name rnnt-speech-recognition \
    --cap-add SYS_ADMIN \
    --gpus <gpus> \
    -v <path to local data>:/rnnt-speech-recognition/data \
    -v <path to save model locally>:/rnnt-speech-recognition/model \
    -e MODE=train \
    -e DATA_DIR=./data \
    -e MODEL_DIR=./model \
    noahchalifour/rnnt-speech-recognition
```