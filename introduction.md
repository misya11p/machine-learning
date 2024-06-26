# 機械学習

機械学習ってなんだぁ？

## 機械学習とは

*Machine Learning, ML*

コンピュータがデータを学習すること。

データを学習するって具体的に何やねん、という感じだが、<u>何らかの値をデータから何かいい感じに求めること</u>で、正直これ以上の説明はない。調べても大体こんな感じのことがふわっと書いてあるだけだと思う。

なぜこのような抽象的な説明になるかというと、それは「機械学習」という言葉がとても広い意味を持っているからである。広い意味を持つ言葉の一般的な説明が抽象的になるのは必然である。ただ、個人的にはもう少し具体的な説明ができると思っている。その説明は多少一般性に欠けるが、本書で学ぶ大体のモデルに当てはまるので、今からその説明をしよう。

<br>

機械学習とは、数理モデルにおける**パラメータ**をデータから求めることである。

数理モデルとはなんらかの現象を数学的に記述したもので、ざっくりと「数式」とか「計算方法」と認識して良い。機械学習では入力と出力を考える場合も多いので、初めは「関数」と認識してもいい。また、パラメータとは数理モデルが演算に用いる値で、例えば$f(x)=ax+b$というモデルでは$a$と$b$がパラメータにあたる。このパラメータをデータから求めることを機械学習といい、また機械学習を活用したモデルを機械学習モデルと呼ぶ。

また、機械学習で求めるのは演算に使用する値（=パラメータ）で、演算方法ではないことに注意。演算方法は人間が与える。先の例だと、$f(x)=ax+b$という演算方法は人間が与え、$a$と$b$の具体的な値を機械学習で求める。

<br>

機械学習モデルはAI（人工知能）と呼ばれることも多い。ただ<u>AI=機械学習モデル</u>という訳ではないので注意。AIとは「人工的に作った人間の知能」のことで、そこに機械学習モデルも含まれるというだけ。機械学習を用いないAIも存在する。
