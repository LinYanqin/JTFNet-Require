#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimensions
xyz2pipe -in ../3DHNCOfull.proc1/XYZA/ft4sp.xyza.2D                                \
| nmrPipe  -fn ZTP                                  \
#| nmrPipe  -fn LP -fb -pred 64                     \
| nmrPipe  -fn SP -off 0.48 -end 0.95 -pow 2 -c 0.5 \
| nmrPipe  -fn ZF -size 64			                    \
| nmrPipe  -fn FT                                   \
| nmrPipe  -fn PS -hdr                              \
| nmrPipe  -fn PS -p0  0.0 -p1 0 -di  \
| nmrPipe  -fn ZTP                                  \
| nmrPipe  -fn TP -auto                             \
#| nmrPipe  -fn LP -fb -pred 20                     \
| nmrPipe  -fn SP -off 0.48 -end 0.95 -pow 2 -c 0.5 \
| nmrPipe  -fn ZF -size 64		                        \
| nmrPipe  -fn FT                                   \
| nmrPipe  -fn PS -hdr                              \
| nmrPipe  -fn PS -p0 0.0 -p1 -0.0    \
#| nmrPipe  -fn TP  -auto                            \
| pipe2xyz -out ok2.ft -verb -ov

exit
    
