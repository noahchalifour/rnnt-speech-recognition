FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update -y
RUN apt-get install gcc-4.8 g++-4.8 cmake ffmpeg libasound-dev wget -y
RUN wget http://www.portaudio.com/archives/pa_stable_v190600_20161030.tgz
RUN tar xvfz pa_stable_v190600_20161030.tgz
WORKDIR /portaudio
RUN ./configure && make
RUN make install

WORKDIR /rnnt-speech-recognition
COPY . .

RUN pip install -r requirements.txt
RUN chmod +x ./scripts/*

RUN ./scripts/build_rnnt.sh

CMD [ "./scripts/docker_common_voice.sh" ]