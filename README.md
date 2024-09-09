[`SurarueN`](https://github.com/kntt32/SurarueN/tree/main)へ移行済み  

# FullarueN (NeuralNet)  
自作の分類問題向け機械学習ライブラリである  
単純なニューラルネットワークを利用した機械学習ができる  

`neuralnet.c`に基本的なニューラルネットワークのコードを配置している  
コンパイル時のバイナリ設定は`FullarueN_Build.h`のマクロで行う


## 依存ライブラリ
以下のライブラリに依存している：
- ホスト環境C言語標準ライブラリ
- 自作の行列計算ライブラリ [`FlexirtaM`](https://github.com/kntt32/FlexirtaM)

## 使い方
ビルドは`make buildlib`で行う

このライブラリを使用するには
- インクルードパスに`FullarueN.h`及び`FullarueN_Build.h`の配置
- ソース中で`FullarueN_Build.h`をインクルード
- `FullarueN.so`のリンク

が必要である
