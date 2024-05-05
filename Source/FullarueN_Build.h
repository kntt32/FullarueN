#pragma once

#define NEURALNET_BASE_NUMBER_TYPE float
    //数値の内部表記を設定
#define NEURALNET_BASE_NUMBER_CONVERT_OPARATER "%f"
    //BASE_NUMBER_TYPE型の変換演算子
#define NEURALNET_ENABLE_RDRAND 1
    //x64RdRand命令を使用
#define NEURALNET_ENABLE_PTHREAD 1
    //pthreadを許可
#define NEURALNET_ENABLE_WINAPI 0
    //Win32APIを許可する

#include <FullarueN.h>
