import pickle
import numpy as np
import math
import csv
import utils
import train_model
import estimate_price as est
import matplotlib.pyplot as plt
from pprint import pprint

def plot_estimations(data, estimations, labelx, labely):
	# We draw red dots for the dataset
	plt.plot(data.tab_x, data.tab_y, 'ro')
	# And blue dots for the estimations of our model
	plt.plot(data.tab_x, estimations, 'bo')
	# And a dotted line for the price mean
	plt.plot(data.sorted_tab_x, data.mean_y, '--')
	plt.xlabel(labelx)
	plt.ylabel(labely)
	plt.show()

def calculate_sse(data, estimations):
	residuals = []
	squared_residuals = []
	for i in range(len(data.tab_y)):
		residual = estimations[i] - data.tab_y[i]
		residuals.append(residual)
		squared_residuals.append(math.pow(residual, 2))
	sse = sum(squared_residuals)
	return (sse / len(data.tab_y))

def test_progression(data, loops, ratio):
	for i in range(loops):
		data.norm_theta0, data.norm_theta1 = train_model.train_model(ratio, data.norm_theta0, data.norm_theta1, data.norm_tab_x, data.norm_tab_y)
	data.theta0 = utils.de_normalize(data.norm_theta0, data.max_y, data.min_y)
	data.theta1 = utils.de_normalize(data.norm_theta1, data.max_y, data.min_y) / (data.max_x - data.min_x)
	estimations = []
	for x in data.tab_x:
		estimations.append(est.estimate_price(x, data.theta0, data.theta1))
	print("SSE: ", calculate_sse(data, estimations))
	plot_estimations(data, estimations, 'km', 'price')

def main():
	tmp_tab_x, tmp_tab_y = utils.parse_csv('data.csv', 'km', 'price')
	if (tmp_tab_x is None or tmp_tab_y is None):
		return
	data = train_model.Dataset(tmp_tab_x, tmp_tab_y)

	for i in range(5, 100, 5):
		print(i, "iterations")
		test_progression(data, i, 1)
		data.theta0, data.norm_theta0, data.theta1, data.norm_theta1 = 0, 0, 0, 0
		print("")

	data.theta0, data.theta1 = utils.retrieve_thetas()

if __name__ == '__main__':
	main()
