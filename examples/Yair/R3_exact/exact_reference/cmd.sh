   python ../FFT_3D.py -file1 exact_reference/acf_R3_full.dat -tau 13 -dt 0.5 -n1 101  -w2 1  -t Quantum -beta 8 -factor 1 -lmax 1000  &
   python ../FFT_2D.py -beta 8 -tau 13 -dt 0.5 -n1 101 -n2 101 -t Quantum -file1 exact_reference/acf_R3_t0.dat &
