#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimension
set I = 1
echo $I
while ($I < 11)
	set FID_file = ../Rec_Data/Rec_nmrpipe/res_full$I.dat
	xyz2pipe -in $FID_file            \
	| nmrPipe  -fn TP -auto           \
	| nmrPipe  -fn SP -off 0.78 -end 0.95 -pow 2 -c 0.5 \
	| nmrPipe  -fn ZF -size 128		\
	| nmrPipe  -fn FT          \
	| nmrPipe  -fn PS -p0 0.0 -p1 -0.0  -di\
	| nmrPipe  -fn TP  -auto                            \
	| nmrPipe  -fn ZTP                                  \
	| nmrPipe  -fn SP -off 0.78 -end 0.95 -pow 2 -c 0.5 \
	| nmrPipe  -fn ZF -size 128			                    \
	| nmrPipe  -fn FT                    \
	| nmrPipe  -fn PS -p0  0.0 -p1 0 -di\
	| nmrPipe  -fn 	ZTP   \
	| nmrPipe  -fn TP -auto           \
	| pipe2xyz -out ../Rec_Data/Rec_nmrpipe/resCN3D$I.dat -x -verb -ov
	echo $I
	@ I += 1
end
set FID_file = ../Rec_Data/Rec_nmrpipe/ale_full.dat
xyz2pipe -in $FID_file            \
| nmrPipe  -fn TP -auto           \
| nmrPipe  -fn SP -off 0.78 -end 0.95 -pow 2 -c 0.5 \
| nmrPipe  -fn ZF -size 128		\
| nmrPipe  -fn FT          \
| nmrPipe  -fn PS -p0 0.0 -p1 -0.0  -di\
| nmrPipe  -fn TP  -auto                            \
| nmrPipe  -fn ZTP                                  \
| nmrPipe  -fn SP -off 0.78 -end 0.95 -pow 2 -c 0.5 \
| nmrPipe  -fn ZF -size 128			                    \
| nmrPipe  -fn FT                    \
| nmrPipe  -fn PS -p0  0.0 -p1 0 -di\
| nmrPipe  -fn 	ZTP   \
| nmrPipe  -fn TP -auto           \
| pipe2xyz -out ../Rec_Data/Rec_nmrpipe/ale3D.dat -x -verb -ov
set I = 1
echo $I
while ($I < 11)
	proj3D.tcl -in ../Rec_Data/Rec_nmrpipe/resCN3D$I.dat -abs

	nmrPipe -in C13.N15.dat                          \
	| nmrPipe  -ov -out  ../Rec_Data/Rec_nmrpipe/resCN$I.dat
	echo $I
	@ I += 1
end
proj3D.tcl -in ../Rec_Data/Rec_nmrpipe/ale3D.dat -abs

