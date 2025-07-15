# RNN-Transducer Speech Recognition

End-to-end speech recognition using RNN-Transducer in Tensorflow 2

## Overview

This speech recognition model is based off Google's [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/pdf/1811.06621.pdf) research paper and is implemented in Python 3 using Tensorflow 2

## Setup Your Environment

To setup your environment, run the following command:

```bash
git clone https://github.com/noahchalifour/rnnt-speech-recognition.git
cd rnnt-speech-recognition
pip install -r requirements.txt
```

## Common Voice

You can find and download the Common Voice dataset [here](https://commonvoice.mozilla.org/en/datasets)

### Convert all MP3s to WAVs

Before you can train a model on the Common Voice dataset, you must first convert all the audio mp3 filetypes to wavs. Do so by running the following command:

> **_NOTE:_** Make sure you have `ffmpeg` installed on your computer, as it uses that to convert mp3 to wav

```bash
./scripts/common_voice_convert.sh \
  data/cv/en/clips \       # Directory of the clips
  4                        # Number of threads
```

Then you can run the following script to remove missing samples

```bash
python scripts/remove_missing_samples.py \
    --data_dir <data_dir> \
    --replace_old
```

`data_dir` here refers to the `clips` directory inside the downloaded Common Voice data directory.

### Preprocessing dataset

After converting all the mp3s to wavs you need to preprocess the dataset, you can do so by running the following command:

```
python preprocess_common_voice.py \
    --data_dir <data_dir> \
    --output_dir <preprocessed_dir>
```

### Training a model

<!-- #### Training on Host -->

To train a simple model, run the following command:

```
python run_rnnt.py \
    --mode train \
    --data_dir <path to data directory>
```

<!-- #### Training in Docker Container

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
    -e OUTPUT_DIR=./model \
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
    -e OUTPUT_DIR=./model \
    noahchalifour/rnnt-speech-recognition
``` -->
