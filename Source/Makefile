build: main.o neuralnet.o
	gcc  $^ libFlexirtaM.so -lm -Wall -Wextra -O2

install: libFullarueN.so
ifeq ($(OS), Windows_NT)
else
	cp $^ /usr/lib
endif

buildlib: libFullarueN.so

debug: main.c neuralnet.c
	gcc -g $^ libFlexirtaM.so -lm -O2

main.o: main.c FullarueN_Build.h
neuralnet.o: neuralnet.c FullarueN_Build.h

libFullarueN.so: neuralnet.o
	gcc -o libFullarueN.so $^ libFlexirtaM.so -shared -fPIC

%.o: %.c
	gcc -I include -c -Wall -Wextra $<

clean:
ifeq ($(OS), Windows_NT)
	del *.o *.a *.exe
else
	rm *.o *.a *.out
endif
