from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

#seed값 생성

seed = 0   # 일정한 결과를 얻기위해. 단 텐서플로우를 구동시키는 cuDNN 등의 내부 소프트웨어가 자체적으로 랜덤 테이블을 생성하기에 여러번 구동시켜 평균을 내는게 좋다.
numpy.random.seed(seed)
tf.set_random_seed(seed)

dataset = numpy.loadtxt("../dataset/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X,Y, epochs=200, batch_size=10)

print("\n Accuracy: %.4f"%(model.evaluate(X,Y))[1])



