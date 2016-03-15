#!/bin/bash

THEANO_FLAGS=device=cpu,floatX=float32 \
    python -m ipdb main.py
