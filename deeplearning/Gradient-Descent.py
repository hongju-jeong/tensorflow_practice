import tensorflow as tf

data = [[2,81],[4,93],[6,91],[8,97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

learning_rate = 0.1

a = tf.Variable(tf.random_uniform([1],0,10,dtype = tf.float64,seed = 0)) #0에서 10사이 임의의 수 1개 만들어라. seed : 실행시 같은 값이 나올 수 있게 설정
b = tf.Variable(tf.random_uniform([1],0,100, dtype = tf.float64, seed = 0))

y = a * x_data + b

rmse = tf.sqrt(tf.reduce_mean(tf.square(y-y_data)))

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess: #session 함수를 이용해 구동에 필요한 리소스를 컴퓨터에 할당하고 이를 실행시킬 준비를 한다.
    #변수초기화
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(gradient_decent)
        if step % 100 == 0 :
            print("step %.f, RMSE = %.4f, 기울기 a = %.4f y 절편 b = %.4f" % (step, sess.run(rmse), sess.run(a), sess.run(b)))

