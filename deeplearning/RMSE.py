import numpy as np

ab = [3,76]

data = [[2,81],[4,93],[6,91],[8,97]]

x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    return ab[0]*x + ab[1]

def rmse(p,y):
    return np.sqrt(((p-y)**2).mean())   #  mean함수를 사용하므로 p와 y의 자리에는 리스트가 들어갈 것이라 예측이 가능하다.

def rmse_val(predict_result, y):
    return rmse(np.array(predict_result),np.array(y))  # np.array를 알아보자

predict_result = []

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한 시간=%.f, 실제 점수=%.f, 예측 점수=%.f" % (x[i], y[i], predict(x[i])))  #파이썬 print 함수 문자열 포매팅

print("rmse 최종값:", rmse_val(predict_result,y))


