# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리를 불러옵니다.
import numpy
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 준비된 수술 환자 데이터를 불러들입니다.
Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장합니다.
X = Data_set[:,0:17]
Y = Data_set[:,17]

# 딥러닝 구조를 결정합니다(모델을 설정하고 실행하는 부분입니다).
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu')) # activation 다음 층으로 어떻게 값을 넘길지 결정하는 부분. 여기서는 가장 많이 사용되는 relu와 sigmoid 함수를 사용하게끔 지정함
model.add(Dense(1, activation='sigmoid'))

# 딥러닝을 실행합니다.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) # loss 한 번 신경망이 실행될 대마다 오차 값을 추적하는 함수. optimizer 오차를 어떻게 줄여 나갈지 정하는 함수.
model.fit(X, Y, epochs=30, batch_size=10)                                           # ㄴ metrics 함수 : 모델이 컴파일될 때 모델 수행 결과를 나타내게끔 설정하는 부분. 정확도를 측정하기 위해 사용되는 테스트 샘플을 학습 과정에서 제외시킴으로써 과적합 문제(특정 데이터에서는 잘 작동하나 다른 데이터에서는 잘 작동하지 않는 문제)를 방지하는 기능
#epoch(에포크) : 전체 샘플 재사용이 30회가 될때 까지 반복    // batch_size : 숫자 만큼 끊어서 집어넣어
# 결과를 출력합니다.
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
