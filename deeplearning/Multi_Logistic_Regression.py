import tensorflow as tf
import numpy as np

seed = 0       #실행할 때마다 같은 결과를 출력하기 위한 seed값 설정
np.random.seed(seed)
tf.set_random_seed(seed)

x_data = np.array([[2,3],[4,3],[6,4],[8,6],[10,7],[12,8],[14,9]]) #공부한 시간, 과외받은 횟수  1x2 행렬에 해당 되겠음.
y_data = np.array([0,0,0,1,1,1,1]).reshape(7,1)  # 7x1 행렬

X = tf.placeholder(tf.float64, shape=[None, 2])  #None은 행의 수가 정해지지 않았다는 뜻
Y = tf.placeholder(tf.float64, shape=[None, 1])

a = tf.Variable(tf.random_uniform([2,1],dtype=tf.float64))    # 2x1 행렬
b = tf.Variable(tf.random_uniform([1],dtype=tf.float64))

y = tf.sigmoid(tf.matmul(X,a) + b)  # a1*x1 + a2*x2 + b 행렬의 곱

loss = -tf.reduce_mean(Y*tf.log(y)+(1-Y)*tf.log(1-y))

learning_rate = 0.1

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y>0.5, dtype=tf.float64)  # tensor의 자료형을 변환 시켜 주는건데 y>0.5는 아직 잘 모르겠음. 단순히 조건문??
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float64))   # equal 예측값과 같으면 true 반환

with tf.Session() as sess: #session 함수를 이용해 구동에 필요한 리소스를 컴퓨터에 할당하고 이를 실행시킬 준비를 한다.
    #변수초기화
    sess.run(tf.global_variables_initializer())
    for step in range(3001):
        a_, b_, loss_, _ = sess.run([a,b,loss,gradient_decent], feed_dict={X: x_data, Y: y_data})
        if step % 300 == 0 :
            print("step %d, a1=%.4f, a2 = %.4f b = %.4f loss=%.4f" % (step, a_[0], a_[1], b_, loss_))

#실제값 적용하기
    new_x = np.array([7,6.]).reshape(1,2)
    new_y = sess.run(y, feed_dict={X: new_x})

    print("공부한 시간: %d, 과회 수업 횟수: %d"%(new_x[:,0],new_x[:,1]))
    print("합격 가능성: %6.2f %%" %(new_y*100))


