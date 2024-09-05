1. Use NMRPipe to load the NUS data.
2. Run process_direct.com and process_indirect.com in 3D/NMRPipe_Code/ to preprocess the NUS data. Run the script nmrPipe_to_mat.py in 3D/Python_Code/, and convert the preprocessed 3D data to *.mat file.
3. Run 3D/matlab_code/preprocess_NUS_3D.m to generate normalized data.
4. Run test_3D.py in 3D/Python_Code/test_JTF-Net to complete the reconstruction.
5. Run 3D/matlab_code/Restore_hypercomplex.m to restore the reconstructed spectrum to the original hypercomplex format. Then run 3D/Python_Code /FID_to_NMRPipe_temp.py to convert the *.mat file in hypercomplex format into NMRPipe format.
6. Run the 3D/NMRPipe_Code/ recFT.com to project the reconstructed data for display of the reconstructed spectrum. Run 3D/Python_Code/NMRPipe_to_mat_temp.py to convert the already projected data to *.mat file.
7. Run the 3D/matlab_code/prob_predic.m to calculate uncertainty and provide REQUIRER.