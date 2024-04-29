#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <FlexirtaM_Build.h>
#include "FullarueN_Build.h"

#define DEBUG_TOOLARGE(t, n) if(10000 < t || t < -10000) {printf("\nERRIN:%d:%f", __LINE__, n);exit(1);}

static unsigned long long NeuralNet_RandInt();


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



NeuralNet* NeuralNet_Set_LearningTarget(NeuralNet* this, const unsigned int count, const NEURALNET_BASE_NUMBER_TYPE inputs[count*this->inputSize], const NEURALNET_BASE_NUMBER_TYPE target[count*this->outputSize]) {
    if(this == NULL || inputs == NULL || target == NULL) return NULL;

    this->learningTarget.count = count;

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

NeuralNet* NeuralNet_Learn(NeuralNet* this, const unsigned int times, const NEURALNET_BASE_NUMBER_TYPE eta) {
    if(this == NULL) return NULL;

    static const NEURALNET_BASE_NUMBER_TYPE eta2 = 0;

    if(this->batchSize == 0) {
        for(unsigned int learningTimes=0; learningTimes<times; learningTimes++) {
            //勾配を計算
            NeuralNet_Reset_Gradiant(this);
            for(unsigned int i=0; i<this->learningTarget.count; i++) {
                NeuralNet_Set_Delta(this, i);
                NeuralNet_Set_Gradiant(this);
            }

            //重みを変更
            {const unsigned int i=0;
                for(unsigned int k=0; k<this->neuralNet[i].neuronNumber*this->inputSize; k++) {
                    this->neuralNet[i].weight.data[k] -= eta * (this->neuralNet[i].gradiantOfWeight[k] + eta2*this->neuralNet[i].weight.data[k]);
                }
            }

            for(unsigned int i=1; i<this->layerNumber; i++) {
                for(unsigned int k=0; k<this->neuralNet[i].neuronNumber*this->neuralNet[i-1].neuronNumber; k++) {
                    this->neuralNet[i].weight.data[k] -= eta * (this->neuralNet[i].gradiantOfWeight[k] + eta2*this->neuralNet[i].weight.data[k]);
                }
            }
            //バイアスを変更
            for(unsigned int i=0; i<this->layerNumber; i++) {
                for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
                    this->neuralNet[i].bias.data[k] -= eta * (this->neuralNet[i].gradiantOfBias[k] + eta2*this->neuralNet[i].bias.data[k]);
                }
            }
        }
    }else {
        printf("AAAAA");
        for(unsigned int learningTimes=0; learningTimes<times; learningTimes++) {
            //勾配を計算
            NeuralNet_Reset_Gradiant(this);
            for(unsigned int i=0; i<this->batchSize; i++) {
                NeuralNet_Set_Delta(this, NeuralNet_RandInt()%this->batchSize);//coding now...
                NeuralNet_Set_Gradiant(this);
            }

            //重み変更
            {const unsigned int i=0;
                for(unsigned int k=0; k<this->neuralNet[i].neuronNumber*this->inputSize; k++) {
                    this->neuralNet[i].weight.data[k] -= eta * this->neuralNet[i].gradiantOfWeight[k];
                }
            }
            for(unsigned int i=1; i<this->layerNumber; i++) {
                for(unsigned int k=0; k<this->neuralNet[i].neuronNumber*this->neuralNet[i-1].neuronNumber; k++) {
                    this->neuralNet[i].weight.data[k] -= eta * this->neuralNet[i].gradiantOfWeight[k];
                }
            }

            //バイアス変更
            for(unsigned int i=0; i<this->layerNumber; i++) {
                for(unsigned int k=0; k<this->neuralNet[i].neuronNumber; k++) {
                    this->neuralNet[i].bias.data[k] -= eta * this->neuralNet[i].gradiantOfBias[k];
                }
            }
        }
    }

    return this;
}

NeuralNet* NeuralNet_Set_BatchSize(NeuralNet* this, const unsigned int size) {
    if(this == NULL) return NULL;

    this->batchSize = size;

    return this;
}


static unsigned long long NeuralNet_RandInt() {
#if NEURALNET_ENABLE_RDRAND
    unsigned long long result = 0;
    asm("rdrand %0" : "=r"(result));
    return result;
#else
    return rand();
#endif
}

