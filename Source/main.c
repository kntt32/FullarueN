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
    NeuralNet_Set_LearningTarget(nnet, 4, learningInputs, learningTarget);
    NeuralNet_Learn(nnet, 300, 0.1);

    NeuralNet_Print_WeightAndBias(nnet);
    

    for(unsigned int i=0; i<4; i++) {
        NeuralNet_Set_Input(nnet, 2, learningInputs+2*i);
        NeuralNet_Run(nnet);
        printf("\n%d:\n", i);
        NeuralNet_Print_Output(nnet);
    }
    
    NeuralNet_Delete(nnet);

#else
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


#endif

    printf("Program ended\n");
    return 0;
}

