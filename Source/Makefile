build: main.o neuralnet.o FlexirtaM.so
	gcc $^ -lm -Wall -Wextra -O2

install: FlexirtaM.so
ifeq ($(OS), Windows_NT)
else
	cp $^ /usr/lib
endif

buildlib: FullarueN.so	

debug: main.c neuralnet.c FlexirtaM.so
	gcc -g $^ -lm -O2

main.o: main.c FullarueN_Build.h
neuralnet.o: neuralnet.c FullarueN_Build.h

FullarueN.so: neuralnet.o FlexirtaM.so
	gcc -o FullarueN.so -shared -fPIC $^

%.o: %.c
	gcc -I include -c -Wall -Wextra $<

clean:
ifeq ($(OS), Windows_NT)
	del *.o *.a *.exe
else
	rm *.o *.a *.out
endif
