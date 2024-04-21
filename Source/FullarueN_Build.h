#pragma once

#ifdef Matrix_Struct
#undef Matrix_Struct
#endif
#ifdef Matrix_Method
#undef Matrix_Method
#endif

#define NEURALNET_BASE_NUMBER_TYPE float
    //数値の内部表記を設定
#define NEURALNET_BASE_NUMBER_CONVERT_OPARATER "%f"
    //BASE_NUMBER_TYPE型の変換演算子
#define Matrix_Struct Matrix_float
    //型の名称
#define Matrix_Method(name) Matrix_float##_##name
    //メゾッドの名前
#define NEURALNET_ENABLE_MULTITHREAD 1
    //マルチスレッドを使用する 未サポート

#include <FullarueN.h>
