nmrPipe -in test.fid \
| nmrPipe -fn SOL \
| nmrPipe -fn SP -off 0.50 -end 0.95 -pow 2 -elb 0.0 -glb 0.0 -c 0.5 \
| nmrPipe -fn FT -verb \
| nmrPipe -fn PS -p0 -48 -p1 0.0 -di \
| nmrPipe -fn EXT -x1 10.8ppm -xn 5.8ppm -sw \
| nmrPipe -fn TP \
| nmrPipe -fn SP -off 0.50 -end 0.95 -pow 1 -elb 0.0 -glb 0.0 -c 0.5 \
| nmrPipe -fn ZF -size 176  \
| nmrPipe -fn FT -alt -verb \
| nmrPipe -fn PS -p0 0.0 -p1 0.0 \
| nmrPipe -fn TP \
| nmrPipe -fn POLY -auto -verb \
  -out test.ft2 -ov

