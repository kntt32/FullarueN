#define Matrix_BaseNumber MATRIX_BASE_NUMBER_TYPE

//matrix.c
typedef struct {
    unsigned int width;
    unsigned int height;
    Matrix_BaseNumber* data;
    Matrix_BaseNumber** index;
} Matrix_Struct;


//構造体の生成/破棄関連
Matrix_Struct* Matrix_Method(Constructer)(Matrix_Struct* this, const unsigned int width, const unsigned int height);//コンストラクタ　下記New以外で確保したものは利用前にこれに通す。しないと未定義動作
Matrix_Struct* Matrix_Method(Destructer)(Matrix_Struct* this);//デストラクタ　コンストラクタに通したものはこれに通す
Matrix_Struct* Matrix_Method(New)(const unsigned int width, const unsigned int height);//動的領域から構造体を新たに確保する
void Matrix_Method(Delete)(Matrix_Struct* this);//Matrix_Newで確保した構造体を解放する

//行列の操作関連
Matrix_Struct* Matrix_Method(Reset)(Matrix_Struct* this, const unsigned int width, const unsigned int height);//行列のサイズを変更して0だけにする
Matrix_Struct* Matrix_Method(Array)(Matrix_Struct* this, const unsigned int width, const unsigned int height, const Matrix_BaseNumber* array);//行列にMatrix_BaseNumber配列を代入する
Matrix_Struct* Matrix_Method(Copy)(Matrix_Struct* this, const Matrix_Struct* from);//コピー
Matrix_BaseNumber* Matrix_Method(Element)(Matrix_Struct* this, const unsigned int x, const unsigned int y);//行列の要素(x,y)へのポインタを返す 存在しない要素はNULLとする
Matrix_BaseNumber** Matrix_Method()(Matrix_Struct* this);//行列の要素に二次元行列としてアクセスする
Matrix_Struct* Matrix_Method(Ones)(Matrix_Struct* this, const unsigned int width, const unsigned int height);//行列を1にする
Matrix_Struct* Matrix_Method(Fill)(Matrix_Struct* this, const unsigned int width, const unsigned int height, const Matrix_BaseNumber number);//行列をnumberだけにする
Matrix_Struct* Matrix_Method(Eye)(Matrix_Struct* this, const unsigned int size);//単位行列にする
Matrix_Struct* Matrix_Method(Random)(Matrix_Struct* this, const unsigned int width, const unsigned int height);//0以上1以下でランダムにする
Matrix_Struct* Matrix_Method(SignedRandom)(Matrix_Struct* this, const unsigned int width, const unsigned int height);//-1以上1以下でランダムにする
Matrix_Struct* Matrix_Method(Arrange)(Matrix_Struct* this, const unsigned int width, const unsigned int height, const Matrix_BaseNumber start, const Matrix_BaseNumber increase);//連続した数値を入れる
void Matrix_Method(Print)(const Matrix_Struct* this);//表示
char* Matrix_Method(OutStr)(const Matrix_Struct* this, char* buff, unsigned int* buffsize);//文字列出力

//行列計算関連
Matrix_Struct* Matrix_Method(Add)(Matrix_Struct* this, const Matrix_Struct* matrix);//各要素の足し算　ブロードキャストあり
Matrix_Struct* Matrix_Method(AddFast)(Matrix_Struct* this, const Matrix_Struct* mat1, const Matrix_Struct* mat2);//各要素の足し算高速版　ブロードキャストあり
Matrix_Struct* Matrix_Method(Sub)(Matrix_Struct* this, const Matrix_Struct* matrix);//各要素の引き算　ブロードキャストあり
Matrix_Struct* Matrix_Method(SubFast)(Matrix_Struct* this, const Matrix_Struct* mat1, const Matrix_Struct* mat2);//各要素の引き算高速版　ブロードキャストあり
Matrix_Struct* Matrix_Method(Mul)(Matrix_Struct* this, const Matrix_Struct* matrix);//各要素の掛け算　ブロードキャストあり
Matrix_Struct* Matrix_Method(MulFast)(Matrix_Struct* this, const Matrix_Struct* mat1, const Matrix_Struct* mat2);//各要素の掛け算高速版　ブロードキャストあり
Matrix_Struct* Matrix_Method(Div)(Matrix_Struct* this, const Matrix_Struct* matrix);//各要素の割り算　ブロードキャストあり
Matrix_Struct* Matrix_Method(DivFast)(Matrix_Struct* this, const Matrix_Struct* mat1, const Matrix_Struct* mat2);//各要素の割り算高速版　ブロードキャストあり
Matrix_Struct* Matrix_Method(Dot)(Matrix_Struct* this, const Matrix_Struct* matrix);//行列の掛け算
Matrix_Struct* Matrix_Method(DotFast)(Matrix_Struct* this, const Matrix_Struct* mat1, const Matrix_Struct* mat2);//行列の掛け算高速版
Matrix_Struct* Matrix_Method(Transpose)(Matrix_Struct* this);//行列の転置
Matrix_Struct* Matrix_Method(Inverse)(Matrix_Struct* this);//逆行列
Matrix_Struct* Matrix_Method(MinorDeterminant)(Matrix_Struct* this, const unsigned int x, const unsigned int y);//小行列式
Matrix_BaseNumber Matrix_Method(Cofacter)(const Matrix_Struct* this, const unsigned x, const unsigned y);//余因子
Matrix_Struct* Matrix_Method(CofacterMatrix)(Matrix_Struct* this);//余因子行列
Matrix_BaseNumber Matrix_Method(Det)(const Matrix_Struct* this);//行列式

//置換関連
Matrix_Struct* Matrix_Method(Sigma)(Matrix_Struct* this, const unsigned int width, const unsigned int sigmaIndex);//置換(σ())
Matrix_BaseNumber Matrix_Method(Sgn)(const Matrix_Struct* this);//sgn()



//math.c
Matrix_BaseNumber Matrix_Method(Math_Random)();//乱数生成 x64のみサポート
unsigned long long Matrix_Method(Math_Factorial)(unsigned int number);//階上
