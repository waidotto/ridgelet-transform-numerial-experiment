#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import argparse
import yaml
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()

if(args.config is None):
	configfile = input("使用する設定ファイル名(default: ./config/default.yml) > ")
	if(configfile == ''):
		configfile = './config/default.yml'
else:
	configfile = args.config

with open(configfile, 'r+') as f:
	data = yaml.load(f)

print("1) 関数を1回だけ学習してグラフを表示")
print("2) 関数を繰り返し学習し，重みの分布を見る")
print("3) Ridgelet変換/双対Ridgelet変換の数値計算")
print("4) オラクル分布によるサンプリング学習と評価(先に3を実行すること) ")
s = input("Choose number (default: 1) > ")
if(s == ''):
	n = 1
else:
	n = int(s)
assert 1 <= n and n <= 4

os.makedirs('./temp', exist_ok = True)
with open('./temp/function.m', 'w') as f:
	f.write(data['octave'])
with open('./temp/setting.py', 'w') as f:
	f.write(data['python'])

if(n == 1):
	subprocess.run('./src/learn-and-evaluate.py --mode train-once --epoch 1000 --show-hidden-layer-output=true', shell = True, check = True)
elif(n == 2):
	subprocess.run('./src/learn-and-evaluate.py --mode aggregate-weight --epoch 500 --loop 100', shell = True, check = True)
	subprocess.run('gnuplot ./src/plot-aggregated-weight.plt', shell = True, check = True)
elif(n == 3):
	subprocess.run('octave ./src/ridgelet-transform.m', shell = True, check = True)
	subprocess.run('gnuplot -e "func=\'{0}\'" ./src/plot-ridgelet-transform.plt'.format(data['gnuplot']), shell = True, check = True)
elif(n == 4):
	subprocess.run('./src/rejection-sampling.py', shell = True, check = True)
	subprocess.run('./src/learn-and-evaluate.py --mode evaluate --epoch 1000', shell = True, check = True)

