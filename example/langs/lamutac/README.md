# Lamutac
セルフホスティングコンパイラ

> [!WARNING]
> まだ仕様に完全に追い付けて**無い**ので、本番コードはRust製のインタプリタを使うことをお勧めします。

コマンドライン引数でソースファイルのパスを受け取り、そのコードをLLVM IRにコンパイルして出力します。

なお、ビルドにはLLVMのコンポーネントとclangが必要です。ビルドはMakeで自動化できます。
