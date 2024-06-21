1.The 2D 15N-1H HSQC spectrum of the protein GB1 was downloaded from https://www.ibbr.umd.edu/nmrpipe/demo.html.
2.After downloading, run the nmrPipe scripts fid.com and process_protein.com.
3.Run the Python_Code/nmrPipe_to_mat.py to convert the nmrPipe format data to mat format.
4.The Matlab_Code/preprocess directory contains scripts for data preprocessing. The files mask_temp_3p.txt, mask_temp_5p.txt, mask_temp_8p.txt, mask_temp_10p.txt, mask_temp_125.txt, mask_temp_15.txt, and mask_temp_20.txt are sampling templates with rates of 3%, 5%, 8%, 10%, 12.5%, 15%, and 20%, respectively.
5.If you want to evaluate the reconstruction results and assessment under different sampling rates, first run preprocess_for_spectra_SR.m. If you want to evaluate the reconstruction results and assessment under different SNRs, first run preprocess_for_spectra_SNR.m.
6.Run test_sec.py in the Python_Code/test_JTFNet directory to reconstruct the NUS data.
7.Run post_process.m in the Matlab_Code/Post-processing directory to obtain the reconstruction results and the assessment results from Requirer.

