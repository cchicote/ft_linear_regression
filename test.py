import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def parse_csv(km_x, price_y):
	f = open('data.csv', 'r')
	with f:
		reader = csv.DictReader(f)
		for row in reader:
			km_x.append(int(row['km']))
			price_y.append(int(row['price']))

def calculate_mean(sorted_x, sorted_y):
	mean_y = [int(np.mean(sorted_y))] * len(sorted_x)
	print("Sorted_x:", sorted_x, "\nSorted_y:", sorted_y, "\n")
	return mean_y

def calculate_sse(sorted_y, mean_y):
	residuals_price = []
	squared_residuals_price = []
	for i in range(len(sorted_y)):
		residual_price = mean_y[i] - sorted_y[i]
		residuals_price.append(residual_price)
		squared_residuals_price.append(int(math.pow(residual_price, 2)))
	print("Residuals price:", residuals_price, "\n")
	print("Squared residuals price:", squared_residuals_price, "\n")
	sse = sum(squared_residuals_price)
	return sse

def estimate_price(kilometers, theta0, theta1):
	price = theta0 + theta1 * kilometers
	return (price)

def train_model(ratio, theta0, theta1, km_x, price_y):
	m = len(km_x)
	evolutiont0 = []
	evolutiont1 = []
	tmp_t0 = 0 
	tmp_t1 = 0
	for i in range(m):
		tmp_t0 += ratio * (estimate_price(km_x[i], theta0, theta1) - price_y[i])
		tmp_t1 += ratio * (estimate_price(km_x[i], theta0, theta1) - price_y[i]) * km_x[i]
	theta0 -= tmp_t0
	theta1 -= tmp_t1
	evolutiont0.append(theta0)
	evolutiont1.append(theta1)
	print("########", theta0, theta1, tmp_t0, tmp_t1)
	return (theta0, theta1, evolutiont0, evolutiont1)

def main():
	km_x = []
	price_y = []
	mean_y = []

	#parsing
	parse_csv(km_x, price_y)

	#declarations y
	price_y_og = price_y
	price_y = np.array(price_y, dtype=np.float64)
	price_y_max = price_y.max()
	price_y  = price_y / price_y_max

	#declarations x
	km_x_og = km_x
	km_x = np.array(km_x, dtype=np.float64)
	km_x_max = km_x.max()
	km_x = km_x / km_x_max
	print("Km_x:", km_x, "\nPrice_y:", price_y, "\n")

	#sorting y on x
	sorted_x, sorted_y = zip(*sorted(zip(km_x, price_y)))
	
	#training
	t0, t1 = 0, 0
	for i in range(1000):
		t0, t1, et0, et1 = train_model(0.01, t0, t1, km_x, price_y)
	
	#estimations
	estimations = []
	for x in km_x:
		estimations.append(estimate_price(x, t0, t1))
	estimations = np.array(estimations)
	estimations = estimations * price_y_max
	print("Estimations", estimations, "\n")
	
	km_x = km_x_og
	price_y = price_y_og

	mean_y = calculate_mean(km_x, price_y)
	print("Mean_y:", mean_y, "\n")

	sse = calculate_sse(price_y, mean_y)
	print("SSE:", sse, "\n")

	print("Estimation AAPL: [", estimate_price(420000 / km_x_max, t0, t1) * price_y_max) 	

	#plt.plot(et0)
	#plt.show()
	
	#plt.plot(et1)
	#plt.show()

	plt.plot(km_x, estimations, 'bo')
	#plt.plot(sorted_x, sorted_y, 'ro')
	plt.plot(km_x, price_y, 'ro')
	plt.plot(km_x, mean_y, '--')
	plt.ylabel('price')
	plt.xlabel('km')
	plt.show()

main()
