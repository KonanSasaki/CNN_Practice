import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Activation,Flatten #Flattenは4次元配列を1次元にしてくれる
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10
batch_size = 128
epochs = 20

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)


y_train  = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)

model = Sequential()
model.add(Dense(128,activation='relu',input_shape=(32,32,3)))
model.add(Dropout(0.4))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size = (2,2),strides = 2,padding='same'))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size = (2,2),strides = 2,padding='same'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Flatten())#Flattenは4次元配列を1次元にしてくれる
model.add(Dense(num_classes,activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

plot_model(model,"model_mnist.png",show_shapes = True,show_layer_names = True)

model.fit(x_train,y_train,batch_size,epochs)

score = model.evaluate(x_test,y_test,verbose=0)

# 学習曲線や学習の様子を表示
print('Test loss',score[0])
print('Test accuracy',score[1])

model.save('model_cifar.h5')
model.save_weights("weights_cifar.hdf5")
