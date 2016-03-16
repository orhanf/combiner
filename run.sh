#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=.95 \
    python -m ipdb main.py
