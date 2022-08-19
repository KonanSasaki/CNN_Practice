"""
解説用サイト
https://note.com/randr_inc/n/n643873336995
https://child-programmer.com/
https://www.infiniteloop.co.jp/tech-blog/2018/02/learning-keras-06/
https://qiita.com/maruware/items/0a474c6d409b83f4bf52
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import utils as np_utils
import matplotlib.pyplot as plt

batch_size = 128 # 訓練データを128ずつのデータに分けて学習させる
num_classes = 10 # 分類させる数。数字なので10種類
epochs = 20 # 訓練データを繰り返し学習させる数

# 訓練データ(train)とテストデータ(test)を取得する
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# 元のデータは28×28だから、784×1に整形する
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
# データタイプ指定
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# データの正規化(画像の各ピクセルは黒から白までの変化を0~255で表している。これを0~1にするために255で割る)。演算子の左側の値を右側の値で割ったのを左側に代入。+=の割り算ver
x_train /= 255
x_test /= 255
"""
y(目的変数)には0から9の数字が入っている。これをkerasで扱いやすい形(one-hot表現)に変換する
出力結果を[0,0,1,2,0,0,6,0,1,0]のようにでき(2の確率10%,3の確率20%,6の確率60%,8の確率10%,その他0%)と表現できる
ソフトマックス関数が絡んでくる
"""
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

# モデルの作成
# ニューラルネットワークは本来は脳と同様、いろんな方向と結びついているが、人間が作成しやすいように、左から右へのように一列にする
model = Sequential()
# 128(バッチサイズ)×784(1枚の画素数)の入力。隠れ層１で512のニューロンを持ち、活性化関数にはrelu関数を用いる
model.add(Dense(512,activation='relu',input_shape=(128,784)))
# 隠れ層1でドロップアウト率(0.2)   ドロップアウトとは隠れ層のノードをランダムで非活性化(わざと一定割合のノードを使わないように指示することによって、使用したノードの重みを最適化する)
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
# 出力層(ニューロンは10,活性化関数にはソフトマックス)
model.add(Dense(num_classes,activation='softmax'))
# 機械学習モデルの詳細を表示する
model.summary()
# 学習を始める前にcompileメソッドを用いて、どのような学習処理をするか設定する。引数は(最適化アルゴリズム,損失関数,評価関数)
model.compile(optimizer=RMSprop(),loss='categorical_crossentropy',metrics=['accuracy'])
# 訓練データの実行
history = model.fit(x_train,y_train, # 引数は(訓練データの配列,目的変数のデータの配列,バッチサイズ。デフォルトは32,エポック数,0か1か2の数字[0:ログ出力なし,1:ログをプログレスバーで出力,2:エポックごとに1行のログ出力],訓練できたか確認する用のデータ)
  batch_size=batch_size,
  epochs=epochs,
  verbose=1,
  validation_split=0.2,
  validation_data=(x_test,y_test))
# 評価はevaluateで行う
score = model.evaluate(x_test,y_test,verbose=0)

# 学習曲線や学習の様子を表示
print('Test loss',score[0])
print('Test accuracy',score[1])
# 使用する評価関数を指定	
metrics = ['loss', 'accuracy'] 
# グラフを表示するスペースを用意
plt.figure(figsize=(10, 5))  
for i in range(len(metrics)):
 
    metric = metrics[i]
    # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.subplot(1, 2, i+1) 
    # グラフのタイトルを表示
    plt.title(metric)  
    # historyから訓練データの評価を取り出す
    plt_train = history.history[metric]  
    # historyからテストデータの評価を取り出す
    plt_test = history.history['val_' + metric]  
    #軸のラベル設定
    plt.xlabel("epochs")
    plt.ylabel("difference")
    # 訓練データの評価をグラフにプロット
    plt.plot(plt_train, label='training')  
    # テストデータの評価をグラフにプロット
    plt.plot(plt_test, label='test')  
    # ラベルの表示
    plt.legend()  
# グラフの表示   
plt.show()  
#モデルの保存
model.save('model.h5')
#モデルの読み込み
#model = keras.models.load_model('model.h5')
# 375という数字は60000枚の画像のうち訓練用として48000枚を用いて、さらにバッチサイズを128に設定しているから。48000 / 128 = 375