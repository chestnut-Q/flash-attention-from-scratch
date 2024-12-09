
CC = mpicxx
OPT = -Ofast -unroll
CFLAGS = -Wall -DGETTIMEOFDAY -std=c++11 $(OPT) -march=skylake-avx512 -fopenmp
LDFLAGS = -Wall 
LDLIBS = -lpthread -lm -fopenmp

targets =  naive opt sgemm
objects =  benchmark.o naive.o opt.o benchmark_sgemm.o sgemm.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

naive : benchmark.o naive.o
	$(CC) -o $@ $^ $(LDLIBS)
opt : benchmark.o sgemm.o opt.o
	$(CC) -o $@ $^ $(LDLIBS)
sgemm : benchmark_sgemm.o sgemm.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)