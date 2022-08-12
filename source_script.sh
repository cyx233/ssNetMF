#!/bin/bash
echo "Please choose the compiler (icc/gcc)"
read compiler
if [ "$compiler"x = "icc"x ];
then
	source /opt/intel/bin/iccvars.sh intel64
	icc -O3 -mkl -qopenmp -w freigs.c ssNetMF.c matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -o eNetMF
else if [ "$compiler"x = "gcc"x ];
then
	gcc -g -O3 -m64 freigs.c ssNetMF.c matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -Wl,--start-group /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/latest/linux/lib -larpack -liomp5 -lpthread -ldl -lm -fopenmp -w -o ssNetMF
else
	echo "Compiler error!"
fi
fi

