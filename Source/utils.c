#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#include <FlexirtaM_Build.h>
#include "FullarueN_Build.h"


NeuralNet* NeuralNet_Print_Output(NeuralNet* this) {
    Matrix_Method(Print)(&(this->output));
    return this;
}

NeuralNet* NeuralNet_Print_WeightAndBias(NeuralNet* this) {
    for(unsigned int i=0; i<this->layerNumber; i++) {
        printf("Layer: %u\n", i);
        Matrix_Method(Print)(&(this->neuralNet[i].weight));
        printf(",\n");
        Matrix_Method(Print)(&(this->neuralNet[i].bias));
        printf("\n");
    }
    
    return this;
}

NeuralNet* NeuralNet_Print_U(NeuralNet* this) {
    printf("U\n");
    for(unsigned int i=0; i<this->layerNumber; i++) {
        Matrix_Method(Print)(&(this->neuralNet[i].u));
        printf(".");
    }
    printf("\n");
    return this;
}

NeuralNet* NeuralNet_Print_Y(NeuralNet* this) {
    printf("Y:\n");
    for(unsigned int i=0; i<this->layerNumber; i++) {
        Matrix_Method(Print)(&(this->neuralNet[i].y));
        printf(".");
    }
    printf("\n");
    return this;
}

NeuralNet* NeuralNet_Print_Gradiant(NeuralNet* this) {
    if(this == NULL) return NULL;
    printf("Gradiants:\n");

    if(this->layerNumber == 0) return this;
    { const unsigned int i=0;
        printf(".");
        for(unsigned int k=0; k<this->inputSize; k++) {
            printf("     {");
            for(unsigned int j=0; j<this->neuralNet[i].neuronNumber; j++) {
                printf(NEURALNET_BASE_NUMBER_CONVERT_OPARATER", ", this->neuralNet[i].gradiantOfWeight[k*this->neuralNet[i].neuronNumber + j]);
            }
            printf("}\n");
        }
        printf(",");
        printf("     {");
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
            printf(NEURALNET_BASE_NUMBER_CONVERT_OPARATER", ", this->neuralNet[i].gradiantOfBias[k]);
        }
        printf("}\n");
    }
    for(unsigned int i=1; i<this->layerNumber; i++) {
        printf(".");
        for(unsigned int k=0; k<this->neuralNet[i-1].neuronNumber; k++) {
            printf("     {");
            for(unsigned int j=0; j<this->neuralNet[i].neuronNumber; j++) {
                printf(NEURALNET_BASE_NUMBER_CONVERT_OPARATER", ", this->neuralNet[i].gradiantOfWeight[k*this->neuralNet[i].neuronNumber + j]);
            }
            printf("}\n");
        }
        printf(",");
        printf("     {");
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
            printf(NEURALNET_BASE_NUMBER_CONVERT_OPARATER", ", this->neuralNet[i].gradiantOfBias[k]);
        }
        printf("}\n");
    }

    return this;
}

NeuralNet* NeuralNet_Print_Delta(NeuralNet* this) {
    printf("Delta:\n");
    for(unsigned int i=0; i<this->layerNumber; i++) {
        printf("     {");
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
            printf(NEURALNET_BASE_NUMBER_CONVERT_OPARATER", ", this->neuralNet[i].deltaOfWeight[k]);
        }
        printf("}\n");
    }

    return this;
}


unsigned long long NeuralNet_RandInt() {
#if NEURALNET_ENABLE_RDRAND
    unsigned long long result = 0;
    asm("rdrand %0" : "=r"(result));
    return result;
#else
    static unsigned int inited = 0;
    if(!inited) srand((unsigned int)time(NULL));
    return rand();
#endif
}

unsigned int* NeuralNet_Shuffle(unsigned int* buff, const unsigned size) {
    if(buff == NULL || size == 0) return NULL;
    bool* stateFlag = malloc(sizeof(bool)*size);
    for(unsigned int i=0; i<size; i++) stateFlag[i] = 1;

    unsigned int selectedIndexByOnflag = 0;
    unsigned int selectedIndexByAll = 0;

    for(unsigned int i=size; 0<i; i--) {
        selectedIndexByOnflag = 1+NeuralNet_RandInt()%(i);
        
        for(unsigned int k=0; k<size; k++) {
            if(stateFlag[k]) {
                selectedIndexByOnflag--;
            }
            if(selectedIndexByOnflag == 0) {
                selectedIndexByAll = k;
                break;
            }
            
        }
        stateFlag[selectedIndexByAll] = 0;
        buff[i-1] = selectedIndexByAll;
    }

    free(stateFlag);
    return buff;
}

