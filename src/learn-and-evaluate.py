#!/usr/bin/env python3

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from matplotlib import pyplot

import argparse

#活性化関数
def activationFunction(x):
	return F.exp(-F.square(x) / 2.0) #Gauss核(正規分布)を用いる

class SimpleChain(Chain):
	def __init__(self, number_of_hidden_nodes):
		super(SimpleChain, self).__init__()
		with self.init_scope():
			ini = chainer.initializers.Normal(scale = 1.0) #重みの初期値は分散が1.0の正規分布に従う
			self.l1 = L.Linear(1, number_of_hidden_nodes, initialW = ini, initial_bias = ini)
			self.l2 = L.Linear(number_of_hidden_nodes, 1, initialW = ini, nobias = False) #出力時はバイアス(定数項)は使用しない

	def __call__(self, x):
		h = activationFunction(self.l1(x))
		return self.l2(h)

class LossChain(Chain):
	def __init__(self, predictor):
		super(LossChain, self).__init__()
		with self.init_scope():
			self.predictor = predictor

	def __call__(self, x, t):
		y = self.predictor(x)
		loss = F.mean_squared_error(y, t)
		report({'loss': loss}, self)
		return loss

#コマンドライン引数解析
parser = argparse.ArgumentParser()
parser.add_argument('--mode', required = True, choices = ['train-once', 'aggregate-weight', 'evaluate'])
parser.add_argument('--epoch', type = int, default = 500)
parser.add_argument('--loop', type = int, default = 100)
parser.add_argument('--show-hidden-layer-output', type = bool, default = False)
args = parser.parse_args()

#設定ファイル読み込み
with open('./temp/setting.py', 'r') as f:
	exec(f.read())

data_axis = np.linspace(-1, 1, number_of_data, dtype = np.float32).reshape((number_of_data, 1))
#data_value = np.cos(3 * np.pi * data_axis) * np.exp(-4 * np.square(data_axis))
data_value = f(data_axis)

def plotGraph(model, show_hidden_layer = False):
	pyplot.plot(data_axis, data_value, label = 'data')

	x_axis = np.linspace(-1, 1, number_of_data, dtype = np.float32).reshape((number_of_data, 1))
	y_axis = model.predictor(x_axis)
	x = x_axis.reshape((number_of_data,))
	y = y_axis.data.reshape((number_of_data,))
	pyplot.plot(x, y, label = 'predict')

	if(show_hidden_layer):
		for i in range(number_of_hidden_nodes):
			l1_value = activationFunction(model.predictor.l1(x_axis)).data[:, i]
			pyplot.plot(x_axis, l1_value, label = 'l1-' + str(i))

	ax = pyplot.gca()
	ax.grid()

	pyplot.legend()
	pyplot.show()

def train(model, number_of_epoch, once = False, show_hidden_layer = False):
	#optimizer = optimizers.SGD()
	#optimizer = optimizers.SGD(lr = 0.5)
	optimizer = optimizers.Adam(alpha = 0.1)
	#optimizer = optimizers.MomentumSGD(lr = 0.1, momentum = 0.9)
	optimizer.setup(model)

	train = datasets.TupleDataset(data_axis, data_value)
	train_iter = iterators.SerialIterator(train, batch_size = number_of_data, shuffle = True)

	updater = training.StandardUpdater(train_iter, optimizer)
	trainer = training.Trainer(updater, (number_of_epoch, 'epoch'), out = 'result')

	if(once):
		trainer.extend(extensions.LogReport())
		trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))

	p = model.predictor
	if(not once):
		with open("./output/initial-weight.txt", mode = 'a') as f:
			for (a, b) in zip(p.l1.W.data.reshape((number_of_hidden_nodes,)), p.l1.b.data):
				f.write("{0} {1}\n".format(a, b))

	trainer.run()

	if(not once):
		with open("./output/final-weight.txt", mode = 'a') as f:
			for (a, b) in zip(p.l1.W.data.reshape((number_of_hidden_nodes,)), p.l1.b.data):
				f.write("{0} {1}\n".format(a, b))

	if(once):
		plotGraph(model, show_hidden_layer)
	return

if(args.mode == 'train-once'):
	train(LossChain(SimpleChain(number_of_hidden_nodes)), args.epoch, once = True, show_hidden_layer = args.show_hidden_layer_output)
elif(args.mode == 'aggregate-weight'):
	for i in range(args.loop):
		print("{0}-th train ({0}/{1})".format(i + 1, args.loop))
		train(LossChain(SimpleChain(number_of_hidden_nodes)), args.epoch)
elif(args.mode == 'evaluate'):
	model = LossChain(SimpleChain(number_of_hidden_nodes))
	weight = np.fromregex('./output/oracle-sampled-weight.txt', r"(\S+) (\S+) (\S+)", dtype = [('a', np.float32), ('b', np.float32), ('c', np.float32)])
	a = weight['a'].reshape(number_of_hidden_nodes, 1)
	b = weight['b']
	c = weight['c'].reshape(1, number_of_hidden_nodes)
	model.predictor.l1.W.data = a
	model.predictor.l1.b.data = -b
	model.predictor.l2.W.data = c

	plotGraph(model, show_hidden_layer = args.show_hidden_layer_output)

	train(model, args.epoch, once = True, show_hidden_layer = args.show_hidden_layer_output)


