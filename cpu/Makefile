
CC = mpicxx
OPT = 
CFLAGS = -Wall -DGETTIMEOFDAY -std=c++11 $(OPT)
LDFLAGS = -Wall 
LDLIBS = -lpthread -lm

targets =  naive opt 
objects =  benchmark.o naive.o opt.o 

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

naive : benchmark.o naive.o
	$(CC) -o $@ $^ $(LDLIBS)
opt : benchmark.o opt.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)