#!/bin/csh -f

echo '|   Processing JTF-Net reconstruction '
echo
echo Processing YZ dimensions
xyz2pipe -in ./temp.ft1                     \
| nmrPipe  -fn TP -auto           \
| nmrPipe  -fn ZF -size 64		\
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5\
| nmrPipe  -fn TP  -auto             \
| nmrPipe  -fn ZTP                                  \
| nmrPipe  -fn ZF -size 64	  \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5 \
| nmrPipe  -fn 	ZTP   \
| pipe2xyz -out input.dat -x -verb -ov
exit

