#!/bin/csh -f

echo Processing YZ dimensions
xyz2pipe -in fid/test%03d.fid -x -verb              \
| nmrPipe  -fn ZF -auto                             \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5 \
| nmrPipe  -fn FT -auto                                \
| nmrPipe  -fn PS -p0 -94 -p1 0.0 -di              \
| nmrPipe  -fn EXT -x1 10.8ppm -xn 5.8ppm -sw -verb \
| pipe2xyz -out data/test%03d.ft1 -y -ov -verb      \

xyz2pipe  -in data/test%03d.ft1 -y -verb            \
  > ./temp.ft1


exit
    
