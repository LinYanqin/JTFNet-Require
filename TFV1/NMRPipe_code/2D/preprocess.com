#!/bin/csh

nmrPipe -in test.fid                          \
| nmrPipe  -fn SP -off 0.5 -end 0.8 -pow 1 -c 0.5 \
| nmrPipe  -fn FT -auto                                \
| nmrPipe  -fn PS -p0 -48 -p1 0.0 -verb  -di  \
| nmrPipe -fn EXT -x1 10.8ppm -xn 5.8ppm -sw \
| nmrPipe -fn POLY -auto \
| nmrPipe  -fn TP                                       \
| nmrPipe  -ov -out full.ft1 

nmrPipe -in full.ft1 \
| nmrPipe  -fn EM -lb 15 -c 0.5 \
| nmrPipe  -fn ZF -size 176                     \
| nmrPipe  -ov -out input.dat

