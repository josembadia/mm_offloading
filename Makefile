# the compiler to use
CC = clang

# compiler flags:
OMPFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_53
CFLAGS  = -O2 $(OMPFLAGS) -w -Wsource-uses-openmp
  
#files to link:
LFLAGS =  -liomp5  -Wno-all
  
# the name to use for both the target source file, and the output file:
TARGET = mm_hybrid
  
all: $(TARGET)
  
$(TARGET): $(TARGET).c mm_utils.o
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).c mm_utils.o $(LFLAGS)

reposo: reposo.o
	$(CC) $(CFLAGS) -o reposo reposo.c reposo.o $(LFLAGS)

mm_utils.o: mm_utils.c
	$(CC) $(CFLAGS) -c  mm_utils.c -o mm_utils.o

reposo.o: reposo.c
	$(CC) $(CFLAGS) -c  reposo.c -o reposo.o

clean:
	rm -f $(TARGET) *.o
