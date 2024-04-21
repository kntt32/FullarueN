#pragma once

#ifdef MATRIX_BASE_NUMBER_TYPE
#undef MATRIX_BASE_NUMBER_TYPE
#endif
#ifdef Matrix_Struct
#undef Matrix_Struct
#endif
#ifdef Matrix_Method
#undef Matrix_Method
#endif
#ifdef MATRIX_BASE_NUMBER_CONVERT_OPARATER
#undef MATRIX_BASE_NUMBER_CONVERT_OPARATER
#endif
#ifdef MATRIX_ENABLE_RDRAND
#undef MATRIX_ENABLE_RDRAND
#endif
#ifdef MATRIX_ENABLE_MULTITHREAD
#undef MATRIX_ENABLE_MULTITHREAD
#endif

#define MATRIX_BASE_NUMBER_TYPE float
    //数値の内部表記を設定
#define Matrix_Struct Matrix_##float
    //型の名称
#define Matrix_Method(name) Matrix_##float##_##name
    //メゾッドの名前
#define MATRIX_BASE_NUMBER_CONVERT_OPARATER "%f"
    //BASE_NUMBER_TYPE型の変換演算子
#define MATRIX_ENABLE_RDRAND 1
    //RDRAND命令を使用する　x64のみサポート
#define MATRIX_ENABLE_MULTITHREAD 1
    //マルチスレッドを使用する未サポート

#include <FlexirtaM.h>
