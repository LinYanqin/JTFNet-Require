#!/bin/csh -f


echo Processing YZ dimensions
xyz2pipe -in ./fully_sampled/label.dat       \
| nmrPipe  -fn TP -auto           \
| nmrPipe -fn TRI -loc 3 -lHi 2 -rHi 0.5 \
| nmrPipe  -fn ZF -auto		\
| nmrPipe  -fn FT          \
| nmrPipe  -fn PS -p0 0.0 -p1 -0.0  -di\
| nmrPipe  -fn TP  -auto                            \
| nmrPipe  -fn ZTP                                  \
| nmrPipe -fn TRI -loc 3 -lHi 2 -rHi 0.5 \
| nmrPipe  -fn ZF -auto			                    \
| nmrPipe  -fn FT                    \
| nmrPipe  -fn PS -p0  0.0 -p1 0 -di\
| nmrPipe  -fn 	ZTP   \
| nmrPipe  -fn TP -auto           \
| pipe2xyz -out ./res_nmrpipe/full_spec.ft3 -x -verb -ov
proj3D.tcl -in ./res_nmrpipe/full_spec.ft3 -abs
nmrPipe -in CO.N15.dat                          \
| nmrPipe  -ov -out  ./res_nmrpipe/label_proj.dat
exit

