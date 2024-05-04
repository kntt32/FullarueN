build: main.o neuralnet.o learning.o utils.o
ifeq ($(OS), Windows_NT)
	gcc  $^ libFlexirtaM.so -lm -Wall -Wextra -Ofast
else
	gcc  $^ -lFlexirtaM -lm -Wall -Wextra -Ofast
endif

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
learning.o: learning.c FullarueN_Build.h
utils.o: utils.c FullarueN_Build.h

libFullarueN.so: neuralnet.o learning.o utils.o
ifeq ($(OS), Windows_NT)
	gcc -o libFullarueN.so $^ libFlexirtaM.so -shared -fPIC
else
	gcc -o libFullarueN.so $^ -lFlexirtaM -shared -fPIC
endif

%.o: %.c
	gcc -I include -c -Wall -Wextra -Ofast $<

clean:
ifeq ($(OS), Windows_NT)
	del *.o *.a *.exe
else
	rm *.o *.a *.out
endif
