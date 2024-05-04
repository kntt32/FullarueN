#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#include <FlexirtaM_Build.h>
#include "FullarueN_Build.h"

#define DEBUG_TOOLARGE(t, n) if(10000 < t || t < -10000) {printf("\nERRIN:%d:%f", __LINE__, n);exit(1);}

typedef struct {
    unsigned int neuronNumber;//ニューロン個数
    Matrix_Struct weight;//[w:neuronNumber, h:neuronNumber_-1] 重み
    Matrix_Struct bias;//[w:neuronNumber, h:1] バイアス
    Matrix_Struct u;//[w:neuronNumber, h:1] 活性
    Matrix_Struct y;//[w:neuronNumber, h:1] 出力

    NEURALNET_BASE_NUMBER_TYPE* deltaOfWeight;//[w:neuronNumber] 重みの誤差
    NEURALNET_BASE_NUMBER_TYPE* gradiantOfWeight;//[w:neuronNumber h:neuronNumber_-1] 重みの勾配
    NEURALNET_BASE_NUMBER_TYPE* gradiantOfBias;//[w:neuronNumber h:1] バイアスの勾配
} Dummy_neuralNet;//ニューラルネットワークのダミー
struct {
    unsigned int count;//学習データの個数
    NEURALNET_BASE_NUMBER_TYPE** inputs;//[count][neuralNet[0].neuronNumber] 入力
    NEURALNET_BASE_NUMBER_TYPE** target;//[count][neuralNet[layerNumber-1].neuronNumber] 出力
} Dummy_learningTarget;


NeuralNet* NeuralNet_Constructer(NeuralNet* this, const unsigned int inputSize, const unsigned int layerNum, const unsigned int neuronNumbers[layerNum]) {
    if(this == NULL || neuronNumbers == NULL) return NULL;

    this->layerNumber = layerNum;

    this->batchSize = 0;

    this->inputSize = inputSize;

    Matrix_Method(Constructer)(&(this->input), inputSize, 1);

    this->outputSize = neuronNumbers[layerNum-1];

    Matrix_Method(Constructer)(&(this->output), this->outputSize, 1);

    this->neuralNet = malloc(sizeof(Dummy_neuralNet)*layerNum);
    for(unsigned int i=0; i<layerNum; i++) {
        this->neuralNet[i].neuronNumber = neuronNumbers[i];
        
        Matrix_Method(Constructer)(&(this->neuralNet[i].weight), 0, 0);
        Matrix_Method(SignedRandom_Wide)(&(this->neuralNet[i].weight), neuronNumbers[i], (i != 0)?(neuronNumbers[i-1]):(inputSize), 10);
        
        Matrix_Method(Constructer)(&(this->neuralNet[i].bias), 0, 0);
        Matrix_Method(SignedRandom_Wide)(&(this->neuralNet[i].bias), neuronNumbers[i], 1, 10);

        Matrix_Method(Constructer)(&(this->neuralNet[i].u), neuronNumbers[i], 1);
        Matrix_Method(Constructer)(&(this->neuralNet[i].y), neuronNumbers[i], 1);

        this->neuralNet[i].deltaOfWeight = calloc(neuronNumbers[i], sizeof(NEURALNET_BASE_NUMBER_TYPE));

        this->neuralNet[i].gradiantOfWeight = calloc(neuronNumbers[i]*((i != 0)?(neuronNumbers[i-1]):(inputSize)), sizeof(NEURALNET_BASE_NUMBER_TYPE));
        
        this->neuralNet[i].gradiantOfBias = calloc(neuronNumbers[i], sizeof(NEURALNET_BASE_NUMBER_TYPE));
    }

    this->learningTarget.count = 0;
    this->learningTarget.inputs = NULL;
    this->learningTarget.target = NULL;

    return this;
}


NeuralNet* NeuralNet_New(const unsigned int inputSize, const unsigned int layerNum, const unsigned int neuronNumbers[layerNum]) {
    if(neuronNumbers == NULL) return NULL;
    return NeuralNet_Constructer(malloc(sizeof(NeuralNet)), inputSize, layerNum, neuronNumbers);
}


NeuralNet* NeuralNet_Destructer(NeuralNet* this) {
    if(this == NULL) return NULL;

    free(this->learningTarget.inputs);
    free(this->learningTarget.target);
    this->learningTarget.count = 0;

    for(unsigned int i=0; i<this->layerNumber; i++) {
        Matrix_Method(Destructer)(&(this->neuralNet[i].weight));
        Matrix_Method(Destructer)(&(this->neuralNet[i].bias));
        Matrix_Method(Destructer)(&(this->neuralNet[i].u));
        Matrix_Method(Destructer)(&(this->neuralNet[i].y));
        free(this->neuralNet[i].deltaOfWeight);
        free(this->neuralNet[i].gradiantOfWeight);
        free(this->neuralNet[i].gradiantOfBias);
    }
    free(this->neuralNet);
    this->neuralNet = NULL;

    Matrix_Method(Destructer)(&(this->output));

    this->outputSize = 0;

    Matrix_Method(Destructer)(&(this->input));

    this->inputSize = 0;
    
    this->layerNumber = 0;
    
    return this;
}

NeuralNet* NeuralNet_Delete(NeuralNet* this) {
    NeuralNet_Destructer(this);
    free(this);
    return NULL;
}

NeuralNet* NeuralNet_Set_Input(NeuralNet* this, const unsigned int inputSize, const NEURALNET_BASE_NUMBER_TYPE input[]) {
    if(inputSize < this->inputSize) return NULL;

    for(unsigned int i=0; i<this->inputSize; i++) {
        this->input.data[i] = input[i];
    }

    return this; 
}

NeuralNet* NeuralNet_Run(NeuralNet* this) {
    if(this->layerNumber == 0) return NULL;
    {//一層目
        Matrix_Method(DotFast)(&(this->neuralNet[0].u), &(this->input), &(this->neuralNet[0].weight));
        Matrix_Method(Add)(&(this->neuralNet[0].u), &(this->neuralNet[0].bias));

        if(this->layerNumber != 1) {
            for(unsigned int i=0; i<this->neuralNet[0].neuronNumber; i++) {
                this->neuralNet[0].y.data[i] = (this->neuralNet[0].u.data[i] <= 0)?(0):(this->neuralNet[0].u.data[i]);
            }
        }
    }
    for(unsigned int i=1; i<this->layerNumber; i++) {//二層目以降
        Matrix_Method(DotFast)(&(this->neuralNet[i].u), &(this->neuralNet[i-1].y), &(this->neuralNet[i].weight));
        Matrix_Method(Add)(&(this->neuralNet[i].u), &(this->neuralNet[i].bias));

        if(i == this->layerNumber-1) break;
        for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
            this->neuralNet[i].y.data[k] = (this->neuralNet[i].u.data[k] <= 0)?(0):(this->neuralNet[i].u.data[k]);
        
        }
    }

    {
        if(this->outputSize != 1) {
            NEURALNET_BASE_NUMBER_TYPE maxOfU = this->neuralNet[this->layerNumber-1].u.data[0];
            for(unsigned int i=1; i<this->neuralNet[this->layerNumber-1].neuronNumber; i++) {
                if(maxOfU < this->neuralNet[this->layerNumber-1].u.data[i]) maxOfU = this->neuralNet[this->layerNumber-1].u.data[i];
            }
            NEURALNET_BASE_NUMBER_TYPE softMax_Denominator = 0;
            for(unsigned int i=0; i<this->neuralNet[this->layerNumber-1].neuronNumber; i++) {
                softMax_Denominator += exp(this->neuralNet[this->layerNumber-1].u.data[i] - maxOfU);//////バグの元
            }
            for(unsigned int i=0; i<this->neuralNet[this->layerNumber-1].neuronNumber; i++) {
                    this->neuralNet[this->layerNumber-1].y.data[i] = exp(this->neuralNet[this->layerNumber-1].u.data[i] - maxOfU)/softMax_Denominator;
            }
        }else {
            this->neuralNet[this->layerNumber-1].y.data[0] = 1/(1+exp(-(this->neuralNet[this->layerNumber-1].u.data[0])));
        }
    }
    Matrix_Method(Copy)(&(this->output), &(this->neuralNet[this->layerNumber-1].y));

    return NULL;
}
