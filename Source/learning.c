#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if NEURALNET_ENABLE_WINAPI
#include <windows.h>
#elif NEURALNET_ENABLE_PTHREAD
#include <pthread.h>
#endif

#include <FlexirtaM_Build.h>
#include "FullarueN_Build.h"

NeuralNet* NeuralNet_Set_LearningTarget(NeuralNet* this, const unsigned int count, const NEURALNET_BASE_NUMBER_TYPE inputs[count*this->inputSize], const NEURALNET_BASE_NUMBER_TYPE target[count*this->outputSize]) {
    if(this == NULL || inputs == NULL || target == NULL) return NULL;

    this->learningTarget.count = count;

    this->learningTarget.batchSize = count;

    this->learningTarget.inputs = malloc(sizeof(NEURALNET_BASE_NUMBER_TYPE)*count*this->inputSize);
    for(unsigned int i=0; i<count*this->inputSize; i++) {
        this->learningTarget.inputs[i] = inputs[i];
    }

    this->learningTarget.target = malloc(sizeof(NEURALNET_BASE_NUMBER_TYPE)*count*this->outputSize);
    for(unsigned int i=0; i<count*this->outputSize; i++) {
        this->learningTarget.target[i] = target[i];
    }

    return this;
}



NeuralNet* NeuralNet_Set_Delta(NeuralNet* this, const unsigned int learningIndex) {
    if(this == NULL || this->learningTarget.count <= learningIndex) return NULL;
    NeuralNet_Set_Input(this, this->inputSize, this->learningTarget.inputs + this->inputSize*learningIndex);
    NeuralNet_Run(this);

    {
        const unsigned int i=this->layerNumber-1;
        for(unsigned int k=0; k<this->outputSize; k++) {
            this->neuralNet[i].deltaOfWeight[k] = this->output.data[k] - this->learningTarget.target[learningIndex*this->outputSize + k];
        }
    }
    for(int i=this->layerNumber-2; 0<=i; i--) {
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
            this->neuralNet[i].deltaOfWeight[k] = 0;
            if(this->neuralNet[i].u.data[k] < 0) continue;
            for(unsigned int j=0; j<this->neuralNet[i+1].neuronNumber; j++) {
                this->neuralNet[i].deltaOfWeight[k] += this->neuralNet[i+1].deltaOfWeight[j] * this->neuralNet[i+1].weight.data[this->neuralNet[i+1].weight.width * k + j];
            }
        }
    }

    return this;
}

NeuralNet* NeuralNet_Reset_Gradiant(NeuralNet* this) {
    if(this == NULL) return NULL;
    
    { const unsigned int i=0;
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber*this->inputSize; k++) {
            this->neuralNet[i].gradiantOfWeight[k] = 0;
        }
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
            this->neuralNet[i].gradiantOfBias[k] = 0;
        }
    }
    for(unsigned int i=1; i<this->layerNumber; i++) {
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber*this->neuralNet[i-1].neuronNumber; k++) {
            this->neuralNet[i].gradiantOfWeight[k] = 0;
        }
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
            this->neuralNet[i].gradiantOfBias[k] = 0;
        }
    }

    return this;
}

NeuralNet* NeuralNet_Set_Gradiant(NeuralNet* this) {
    if(this == NULL) return NULL;

    if(this->layerNumber == 0) return this;
    { const unsigned int i=0;
        for(unsigned int k=0; k<this->inputSize; k++) {
            for(unsigned int j=0; j<this->neuralNet[i].neuronNumber; j++) {
                this->neuralNet[i].gradiantOfWeight[j + k*this->neuralNet[i].neuronNumber]
                += this->neuralNet[i].deltaOfWeight[j] * this->input.data[k];
            }
        }
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
            this->neuralNet[i].gradiantOfBias[k] += this->neuralNet[i].deltaOfWeight[k];
        }
    }
    
    for(unsigned int i=1; i<this->layerNumber; i++) {
        for(unsigned int k=0; k<this->neuralNet[i-1].neuronNumber; k++) {
            for(unsigned int j=0; j<this->neuralNet[i].neuronNumber; j++) {
                this->neuralNet[i].gradiantOfWeight[j + k*this->neuralNet[i].neuronNumber]
                += this->neuralNet[i].deltaOfWeight[j] * this->neuralNet[i-1].y.data[k];
            }
        }
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
            this->neuralNet[i].gradiantOfBias[k] += this->neuralNet[i].deltaOfWeight[k];
        }
    }

    return this;
}

NeuralNet* NeuralNet_Learn(NeuralNet* this, const unsigned int times) {
    if(this == NULL) return NULL;

    static const NEURALNET_BASE_NUMBER_TYPE eta2 = 0;

    const NEURALNET_BASE_NUMBER_TYPE eta = this->learningTarget.eta;

    unsigned int* learningOrder = malloc(sizeof(unsigned int) * this->learningTarget.count);

    for(unsigned int epochTime = 0; epochTime < times; epochTime++) {
        NeuralNet_Shuffle(learningOrder, this->learningTarget.count);
        for(unsigned int timesInEpoch = 0; timesInEpoch < this->learningTarget.count/this->batchSize; timesInEpoch++) {
            //勾配を計算
            NeuralNet_Reset_Gradiant(this);
            for(unsigned int i=0; i<this->learningTarget.batchSize; i++) {
                NeuralNet_Set_Delta(this, learningOrder[i+timesInEpoch*this->learningTarget.batchSize]);
                NeuralNet_Set_Gradiant(this);
            }

            //重みを更新
            {const unsigned int targetLayerIndex = 0;
                for(unsigned int i=0; i<this->neuralNet[targetLayerIndex].neuronNumber*this->inputSize; i++) {
                    this->neuralNet[targetLayerIndex].weight.data[i] -= eta*(this->neuralNet[targetLayerIndex].gradiantOfWeight[i] + eta2*this->neuralNet[targetLayerIndex].weight.data[i]);
                }
            }
            for(unsigned int targetLayerIndex = 1; targetLayerIndex < this->layerNumber; targetLayerIndex++) {
                for(unsigned int i=0; i<this->neuralNet[targetLayerIndex].neuronNumber*this->neuralNet[targetLayerIndex-1].neuronNumber; i++) {
                    this->neuralNet[targetLayerIndex].weight.data[i] -= eta*(this->neuralNet[targetLayerIndex].gradiantOfWeight[i] + eta2*this->neuralNet[targetLayerIndex].weight.data[i]);
                }
            }

            //バイアスを更新
            for(unsigned int targetLayerIndex = 0; targetLayerIndex < this->layerNumber; targetLayerIndex++) {
                for(unsigned int i=0; i<this->neuralNet[targetLayerIndex].neuronNumber; i++) {
                    this->neuralNet[targetLayerIndex].bias.data[i] -= eta*(this->neuralNet[targetLayerIndex].gradiantOfBias[i] + eta2*this->neuralNet[targetLayerIndex].bias.data[i]);
                }
            }
        }
    }

    free(learningOrder);
    learningOrder = NULL;
    return this;
}

NeuralNet* NeuralNet_Set_BatchSize(NeuralNet* this, const unsigned int size) {
    if(this == NULL) return NULL;

    this->learningTarget.batchSize = size;

    return this;
}

NeuralNet* NeuralNet_Set_Eta(NeuralNet* this, const NEURALNET_BASE_NUMBER_TYPE eta) {
    if(this == NULL) return NULL;

    this->learningTarget.batchSize = eta;
    
    return this;
}
