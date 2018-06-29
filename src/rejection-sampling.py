#!/usr/bin/env python

#棄却法でオラクル分布から重みを生成する

import numpy as np
import numpy.matlib
import random

#活性化関数
def eta(x):
	return np.exp(-np.square(x) / 2.0) #Gauss関数(正規分布)

with open('./temp/setting.py') as f:
	exec(f.read())

#ファイル読み込み
dist = np.loadtxt('./output/numerical-ridgelet.txt', usecols = range(J))
Max = np.max(np.abs(dist))
Min = np.min(np.abs(dist))

j = 0
tries = 0

a = np.zeros(number_of_hidden_nodes) #a_j
b = np.zeros(number_of_hidden_nodes) #b_j
c = np.zeros(number_of_hidden_nodes) #c_j

#棄却法で(a_j, b_j)を生成する
print('oracle sampling...')
while(True):
	tries += 1
	ra = random.randint(0, I - 1)
	rb = random.randint(0, J - 1)
	r = random.uniform(Min, Max)
	if(r <= abs(dist[ra][rb])):
		a[j] = -30 + ra * Delta_a
		b[j] = -30 + rb * Delta_b
		j += 1
	if(j >= number_of_hidden_nodes):
		print("done. ({0} times tried)".format(tries))
		break

#最小二乗法でc_jを求める
x = np.linspace(-1, 1, N, dtype = np.float32)

xs3d = np.kron(np.ones([number_of_hidden_nodes, number_of_hidden_nodes, 1]), x) #(x_s)_ijs
ai3d = np.kron(np.ones([1, number_of_hidden_nodes, N]), a.reshape(number_of_hidden_nodes, 1, 1)) #(a_i)_ijs
bi3d = np.kron(np.ones([1, number_of_hidden_nodes, N]), b.reshape(number_of_hidden_nodes, 1, 1)) #(b_i)_ijs
aj3d = np.kron(np.ones([number_of_hidden_nodes, 1, N]), a.reshape(1, number_of_hidden_nodes, 1)) #(a_j)_ijs
bj3d = np.kron(np.ones([number_of_hidden_nodes, 1, N]), b.reshape(1, number_of_hidden_nodes, 1)) #(b_j)_ijs
A = np.sum(eta(ai3d * xs3d - bi3d) * eta(aj3d * xs3d - bj3d), axis = 2) #s軸で総和を取る

xs2d = np.matlib.repmat(x, number_of_hidden_nodes, 1) #(x_s)_is
ai2d = np.matlib.repmat(a, N, 1).transpose() #(a_i)_is
bi2d = np.matlib.repmat(b, N, 1).transpose() #(b_i)_is
vec_b = np.sum(f(xs2d) * eta(ai2d * xs2d - bi2d), axis = 1) #s軸で総和を取る

c = np.linalg.inv(A).dot(vec_b) #連立一次方程式を解く

print('writing to ./output/oracle-sampled-weight.txt')
with open("./output/oracle-sampled-weight.txt", mode = 'w') as f:
	for j in range(number_of_hidden_nodes):
		f.write("{0:.6f} {1:.6f} {2:.6f}\n".format(a[j], b[j], c[j]))
print('done.')

