#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, getopt
from matplotlib import pyplot as plt

IDX_EPISODE_NB = 4
IDX_EPISODE_SCORE = 10
IDX_STEPS_NB = 7
IDX_LOSS = 13


def plot_data(path, metric, samples, xlabel='x', ylabel='y', legend='legend'):
	arq = open(path, 'r')

	line = arq.readline()

	x = []
	y = []

	idx = 0
	avg = 0.0
	while line:
		if 'TID:' in line:
			st = line.index('TID:')
			line = line[st:].strip().split(' ')
			gs = int(line[IDX_EPISODE_NB])
			score = float(line[metric])
			avg += score 
			if gs % samples == 0 and gs > 0:
				y.append(avg/samples)
				avg = 0.0
				x.append(idx)
				idx += 1
		line = arq.readline()
	plt.plot(x, y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(legend)
	arq.close()
	plt.show()

def print_help():
	print("train_extract -i <inputfile> -m <metric> -s <samples_number>")
	print("<metric> pode ser 11 ou 14, que significam score e perda (loss), respectivamente.")

path = "train_debug.log"
try:
	opts, args = getopt.getopt(sys.argv[1:], "hi:m:s:")
	path = None
	metric = -1
	samples = 200
	xlabel = 'time step'
	ylabel = 'score'
	for opt, arg in opts:
		if opt=='-h':
			print_help()
		elif opt=='-i':
			path = arg
		elif opt=='-m':
			metric = int(arg)
		elif opt=='-s':
			samples = int(arg)
	if path and metric > 0:
		if metric == 24:
			ylabel='loss'
		plot_data(path, metric, samples, xlabel, ylabel, "")
	else:
		print("ERRO: tente seguir as seguintes instruções")
		print_help()
except getopt.GetoptError:
	print_help()