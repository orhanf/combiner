#!/bin/bash

#THEANO_FLAGS=device=gpu${1},floatX=float32,lib.cnmem=.95 \
THEANO_FLAGS=device=cpu,floatX=float32,lib.cnmem=.95 \
    python  main.py
