#!/bin/csh -f


echo '|   Processing JTF-Net reconstruction '
echo
set I = 1
echo $I
while ($I < 11)
	set FID_file = ../../Python_code/3D/test_code/rec_data/res_full$I.dat
	xyz2pipe -in $FID_file            \
	| nmrPipe  -fn TP -auto           \
	| nmrPipe  -fn ZF -auto		\
	| nmrPipe -fn TRI -loc 3 -lHi 2 -rHi 0.5 \
	| nmrPipe  -fn FT    \
	| nmrPipe  -fn PS -p0 0.0 -p1 -0.0  -di\
	| nmrPipe  -fn TP  -auto                            \
	| nmrPipe  -fn ZTP                                  \
	| nmrPipe  -fn ZF -auto			                    \
	| nmrPipe -fn TRI -loc 3 -lHi 2 -rHi 0.5 \
	| nmrPipe  -fn FT \
	| nmrPipe  -fn PS -p0  0 -p1 0 -di\
	| nmrPipe  -fn 	ZTP   \
	| nmrPipe  -fn TP -auto           \
	| pipe2xyz -out ./res_nmrpipe/res3D$I.dat -x -verb -ov
	echo $I
	@ I += 1
end
set FID_file = ../../Python_code/3D/test_code/rec_data/ale_full.dat
xyz2pipe -in $FID_file            \
| nmrPipe  -fn TP -auto           \
| nmrPipe  -fn ZF -auto		\
| nmrPipe  -fn FT    \
| nmrPipe  -fn PS -p0 0.0 -p1 -0.0  -di\
| nmrPipe  -fn TP  -auto                            \
| nmrPipe  -fn ZTP                                  \
| nmrPipe  -fn ZF -auto			                    \
| nmrPipe  -fn FT  \
| nmrPipe  -fn PS -p0  0 -p1 0 -di\
| nmrPipe  -fn 	ZTP   \
| nmrPipe  -fn TP -auto           \
| pipe2xyz -out ./res_nmrpipe/ale3D.dat -x -verb -ov
set I = 1
echo $I
while ($I < 11)
	proj3D.tcl -in ./res_nmrpipe/res3D$I.dat -abs

	nmrPipe -in CO.N15.dat                          \
	| nmrPipe  -ov -out  ./res_nmrpipe/res_proj$I.dat
	echo $I
	@ I += 1
end
