CC1=mpicc
CC2=gcc
CFLAGS=-fopenmp -lm -O3

laps: laps.c
	$(CC2) laps.c -o laps $(CFLAGS)

lapm: lapm.c
	$(CC1) lapm.c -o lapm $(CFLAGS)

lapm2: lapm2.c
	$(CC1) lapm2.c -o lapm2 $(CFLAGS)

lapm3: lapm3.c
	$(CC2) lapm3.c -o lapm3 $(CFLAGS)

clean:
	rm -f laps lapm lapm2 lapm3