all: render omp opti  coalesced vectorized cuda cuda_2 acc

render: render.c
	gcc -Wall render.c -o render -O3 -lm

omp: render_omp.c
	gcc -Wall render_omp.c -o omp -O3 -lm -fopenmp

opti: render_opti.c
	gcc -Wall render_opti.c -o opti -O3 -lm -fopenmp

cuda: render_cuda.cu
	nvcc render_cuda.cu -o cuda

coalesced: render_coalesced.c
	gcc -Wall render_coalesced.c -o coalesced -O3 -lm

vectorized: render_vectorized.c
	gcc  -Wall render_vectorized.c -o vectorized -O3 -lm -march=native

cuda_2: render_cuda_2.cu
	nvcc render_cuda_2.cu -o cuda_2

acc : render_acc.c
	pgcc render_acc.c -o acc


run: render
	./render 1000

run.short: render
	./render 10

run.long: render
	./render 1000000

clean:
	rm -f render render_omp vectorized opti cuda omp coalesced acc cuda_2
	rm -f *.ppm
	rm -f *.png