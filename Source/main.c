#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <FlexirtaM_Build.h>
#include "FullarueN_Build.h"

int main() {
    printf("hello, world!\n");

#if 0
    static const unsigned int neuronNumber[2] = {10,1};
    static const NEURALNET_BASE_NUMBER_TYPE learningInputs[8] = {0,0,0,1,1,0,1,1};
    static const NEURALNET_BASE_NUMBER_TYPE learningTarget[4] = {0,1,1,0};

    NeuralNet* nnet = NeuralNet_New(2, 2, neuronNumber);
    NeuralNet_Print_WeightAndBias(nnet);
    NeuralNet_Set_LearningTarget(nnet, 4, learningInputs, learningTarget);
    NeuralNet_Learn(nnet, 100, 0.1);

    NeuralNet_Print_WeightAndBias(nnet);
    

    for(unsigned int i=0; i<4; i++) {
        NeuralNet_Set_Input(nnet, 2, learningInputs+2*i);
        NeuralNet_Run(nnet);
        printf("\n%d:\n", i);
        NeuralNet_Print_Output(nnet);
    }
    
    NeuralNet_Delete(nnet);

#elif 0
    #define INPUTSIZE 2
    static const unsigned int neuronNumber[1] = {2};
    static const NEURALNET_BASE_NUMBER_TYPE learningInputs[INPUTSIZE*2] = {0,1, 1,0};
    static const NEURALNET_BASE_NUMBER_TYPE learningTarget[2*2] = {1,0, 0,1};

    NeuralNet* nnet = NeuralNet_New(INPUTSIZE, 1, neuronNumber);
    NeuralNet_Set_LearningTarget(nnet, sizeof(learningInputs)/(sizeof(NEURALNET_BASE_NUMBER_TYPE)*INPUTSIZE), learningInputs, learningTarget);

    NeuralNet_Learn(nnet, 500, 1);

    NeuralNet_Print_WeightAndBias(nnet);

    for(unsigned int i=0; i<2; i++) {
        NeuralNet_Set_Input(nnet, INPUTSIZE, learningInputs+INPUTSIZE*i);
        NeuralNet_Run(nnet);
        printf("\n%d:\n", i);
        NeuralNet_Print_Output(nnet);
    }

    NeuralNet_Delete(nnet);

#else
    static const unsigned int neuronNumber[2] = {100,5};
    static NEURALNET_BASE_NUMBER_TYPE learningInputs[5*100] = {1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,1};
    static NEURALNET_BASE_NUMBER_TYPE learningTarget[5*100] = {0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,1, 1,0,0,0,0};

    static const unsigned int datacount = 30;

    for(unsigned int i=0; i<datacount; i++) {
        for(unsigned int k=0; k<5; k++) {
            learningInputs[i*5+k] = ((i%((unsigned int)pow(2,5-k))) - (i%((unsigned int)pow(2,4-k)))) ? (1):(0);
            learningTarget[i*5+k] = (i%5 == k)?(1):(0);
            printf("%f, ", learningTarget[i*5+k]);
        }
        printf("\n");
    }


    NeuralNet* nnet = NeuralNet_New(5, 2, neuronNumber);

    NeuralNet_Set_LearningTarget(nnet, datacount, learningInputs, learningTarget);
    NeuralNet_Learn(nnet, 1000, 0.01);

    for(unsigned int i=0; i<datacount; i++) {
        NeuralNet_Set_Input(nnet, 5, learningInputs+5*i);
        NeuralNet_Run(nnet);
        NeuralNet_Print_Output(nnet);
    }

    NeuralNet_Delete(nnet);

#endif

    printf("Program ended\n");
    return 0;
}

