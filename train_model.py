import pickle
import numpy as np
import math
import csv
import utils
import estimate_price as est
import matplotlib.pyplot as plt

def train_model(ratio, theta0, theta1, km_x, price_y):
	m = len(km_x)
	tmp_t0, tmp_t1 = 0, 0
	for i in range(m):
		tmp_t0 += ratio * (est.estimate_price(km_x[i], theta0, theta1) - price_y[i])
		tmp_t1 += ratio * (est.estimate_price(km_x[i], theta0, theta1) - price_y[i]) * km_x[i]
	theta0 -= tmp_t0
	theta1 -= tmp_t1
	return (theta0, theta1)

def plot_estimations(theta0, theta1, km_x, price_y):
	estimations = []
	for x in km_x:
		estimations.append(est.estimate_price(x, theta0, theta1))
	plt.plot(km_x, price_y, 'ro')
	plt.plot(km_x, estimations, 'bo')
	plt.xlabel('km')
	plt.ylabel('price')
	plt.show()

def main():
	km_x, price_y = [], []
	norm_km_x, norm_price_y = [], []
	km_x_max, price_y_max = 0, 0 
	sorted_x, sorted_y = [], []
	theta0, theta1 = 0, 0
	loops = 0	

	# parsing number of loops
	print("This program trains our program to estimate the price of a car based on a mileage, we need you, the user, to enter a number of trainings that our program will go through:")
	loops = utils.parse_input_int()
	if (loops == -1):
		return
	print("Ok so you need", loops, "loops")

	# parsing csv
	print("Parsing CSV\n")
	km_x, price_y = utils.parse_csv('data.csv', 'km', 'price')
	if (km_x is None or price_y is None):
		return
	print("Km:", km_x, "\n")
	print("Price:",  price_y, "\n")

	# calculate mean
	mean = utils.calculate_mean(km_x, price_y)
	print("MEAN:", mean)

	# sort values
	sorted_x, sorted_y = zip(*sorted(zip(km_x, price_y)))

	# normalize values
	norm_km_x = np.array(km_x, dtype=np.float64)
	km_x_max = norm_km_x.max()
	norm_km_x = utils.normalize(norm_km_x, km_x_max)
	norm_price_y = np.array(price_y, dtype=np.float64)
	price_y_max = norm_price_y.max()
	norm_price_y = utils.normalize(norm_price_y, price_y_max)
	print("Normalized km:", norm_km_x, "\n")
	print("Normalized price:",  norm_price_y, "\n")

	# training
	for i in range(loops):
		theta0, theta1 = train_model(0.01, theta0, theta1, norm_km_x, norm_price_y)
		print(theta0, theta1)
	
	# saving thetas
	print(km_x_max, price_y_max)
	norm_theta0 = theta0 * price_y_max
	norm_theta1 = theta1 * price_y_max / km_x_max
	print("Normalized thetas:", norm_theta0, norm_theta1)
	utils.save_thetas(norm_theta0, norm_theta1)

	plot_estimations(norm_theta0, norm_theta1, km_x, price_y)

	# plot
	plt.plot(km_x, price_y, 'ro')
	plt.plot(sorted_x, mean, '--')
	#plt.show()

if __name__ == '__main__':
	main()
