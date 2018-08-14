import numpy as np

x = [2,4,6,8]
y = [81,93,91,97]

mx = np.mean(x)
my = np.mean(y)

divisor = sum([(i - mx)**2 for i in x])  # 표현방법을 외워두자

def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i]-mx)*(y[i]-my)
    return d

dividend = top(x, mx, y, my)

a = dividend/divisor

b = my - (mx*a)

print("기울기: ", a)
print("y축 절편: ", b)


