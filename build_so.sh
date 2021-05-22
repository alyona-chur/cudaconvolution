#!/bin/bash

echo "Running cudaconvolution shared library building...\n" \
&& nvcc -c cudaconvolution.cu -lrt -lcuda -lcudart -Xcompiler -fPIC -o cudaconvolution.o \
&& g++ -shared cudaconvolution.o -lrt -lcuda -lcudart -L/usr/local/cuda/lib64 -fPIC -o cudaconvolution.so \
&& rm cudaconvolution.o \
&& echo "Shared library is build and ready to be used."


# && cp cudaconvolution.so /usr/lib/ \
# && chmod a-x /usr/lib/cudaconvolution.so \
# && ldconfig -n -v /usr/lib \
