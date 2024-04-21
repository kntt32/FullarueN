#pragma once

/************************************************************
　活性化関数: 隠れ層: ReLU関数
　　　　　　  出力層: ソフトマックス(シグモイド)関数
　　損失関数: クロスエントロピー
************************************************************/

typedef struct {
    unsigned int layerNumber;//レイヤー数

    unsigned int inputSize;//入力サイズ
    Matrix_Struct input;//[w:inputSize h:1] 入力
    unsigned int outputSize;//出力サイズ
    Matrix_Struct output;//[w:outputSize h:1] 出力
    
    struct {
        unsigned int neuronNumber;//ニューロン個数
        Matrix_Struct weight;//[w:neuronNumber, h:neuronNumber_-1] 重み
        Matrix_Struct bias;//[w:neuronNumber, h:1] バイアス
        Matrix_Struct u;//[w:neuronNumber, h:1] 活性
        Matrix_Struct y;//[w:neuronNumber, h:1] 出力

        NEURALNET_BASE_NUMBER_TYPE* deltaOfWeight;//[w:neuronNumber] 重みの誤差
        NEURALNET_BASE_NUMBER_TYPE* gradiantOfWeight;//[w:neuronNumber h:neuronNumber_-1] 重みの勾配
        NEURALNET_BASE_NUMBER_TYPE* gradiantOfBias;//[w:neuronNumber h:1] バイアスの勾配
    }* neuralNet;//[layerNumber] ニューラルネットワーク
    
    struct {
        unsigned int count;//学習データの個数
        NEURALNET_BASE_NUMBER_TYPE* inputs;//[h:count w:neuralNet[0].neuronNumber] 入力
        NEURALNET_BASE_NUMBER_TYPE* target;//[h:count w:neuralNet[layerNumber-1].neuronNumber] 出力
    } learningTarget;//学習データ
} NeuralNet;

NeuralNet* NeuralNet_Constructer(NeuralNet* this, const unsigned int inputSize, const unsigned int layerNum, const unsigned int neuronNumbers[layerNum]);//コンストラクタ 重みとバイアスは-1以上1以下でランダムにセットされる
NeuralNet* NeuralNet_New(const unsigned int inputSize, const unsigned int layerNum, const unsigned int neuronNumbers[layerNum]);//オブジェクト生成
NeuralNet* NeuralNet_Destructer(NeuralNet* this);//デストラクタ
NeuralNet* NeuralNet_Delete(NeuralNet* this);//オブジェクトを破棄

NeuralNet* NeuralNet_Set_LearningTarget(NeuralNet* this, const unsigned int count, const NEURALNET_BASE_NUMBER_TYPE inputs[count*this->inputSize], const NEURALNET_BASE_NUMBER_TYPE target[count*this->outputSize]);//学習データをセット
NeuralNet* NeuralNet_Set_Input(NeuralNet* this, const unsigned int inputSize, const NEURALNET_BASE_NUMBER_TYPE input[]);//入力をセット
NeuralNet* NeuralNet_Run(NeuralNet* this);//実行
NeuralNet* NeuralNet_Print_Output(NeuralNet* this);//結果を表示
NeuralNet* NeuralNet_Print_WeightAndBias(NeuralNet* this);//重みとバイアスを表示
NeuralNet* NeuralNet_Print_U(NeuralNet* this);//Uを表示
NeuralNet* NeuralNet_Print_Y(NeuralNet* this);//Yを表示
NeuralNet* NeuralNet_Print_Delta(NeuralNet* this);//delta(誤差)を表示
NeuralNet* NeuralNet_Print_Gradiant(NeuralNet* this);//勾配を表示

NeuralNet* NeuralNet_Set_Delta(NeuralNet* this, const unsigned int learningIndex);//delta(誤差)をセット
NeuralNet* NeuralNet_Reset_Gradiant(NeuralNet* this);//勾配をリセット
NeuralNet* NeuralNet_Set_Gradiant(NeuralNet* this);//勾配を加算
NeuralNet* NeuralNet_Learn(NeuralNet* this, const unsigned int times, const NEURALNET_BASE_NUMBER_TYPE eta);//学習
