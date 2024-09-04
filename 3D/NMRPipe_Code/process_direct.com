#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimensions
xyz2pipe -in data/test%03d.fid -x -verb              \
| nmrPipe  -fn POLY -time                           \
#| nmrPipe  -fn GM -size 512 -g1 20 -g2 25                     \
| nmrPipe  -fn SP -size 512 -off 0.50 -end 0.8 -pow 2 -c 0.5  \
| nmrPipe  -fn ZF -size 1024                            \
| nmrPipe  -fn FT                                   \
| nmrPipe  -fn PS -p0 0 -p1 0.0 -di              \
| nmrPipe  -fn EXT -x1 11.8ppm -xn 6.8ppm -sw -verb \
| nmrPipe  -fn POLY -auto -ord 0                    \
| pipe2xyz -out data/test%03d.ft1 -y -ov -verb      \

xyz2pipe  -in data/test%03d.ft1 -y -verb            \
  > ./temp.ft1


exit
    
