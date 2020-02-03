import math
import numpy as np
import pickle
import utils

def estimate_price(km, theta0, theta1):
	price = theta0 + theta1 * km
	return (price)

def main():
	theta0, theta1 = 0, 0
	estimated_price = 0
	
	# retrieving thetas
	theta0, theta1 = utils.retrieve_thetas()

	while True:
		# retrieving user input mileage
		km = utils.parse_input_int(utils.bcolors.YELLOW + "Enter a mileage (in km):\n" + utils.bcolors.ENDC)
		if (km == -1):
			return
		
		# estimating price
		# print("Making price estimations with:\nTheta0 =", theta0, "\nTheta1 =", theta1, "\nFor a car with a", km, "km mileage\n")
		estimated_price = estimate_price(km, theta0, theta1)
		print(utils.bcolors.GREEN + "Mileage:" + utils.bcolors.ENDC, " [", km, "]", utils.bcolors.GREEN + "\nPrice:" + utils.bcolors.ENDC, " [", estimated_price, "]\n")

if __name__ == '__main__':
	main()
