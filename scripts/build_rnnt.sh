cd warp-transducer

mkdir build
cd build

CC=gcc-4.8 CXX=g++-4.8 cmake -DCUDA_TOOLKIT_ROOT_DIR=/tmp ..
make
cd ../tensorflow_binding

python setup.py install
cd ../../
