import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np

class bcolors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

class Error(Exception):
	pass

class NegativeValueException(Error):
	pass

def parse_input_int(message=""):
	value = 0
	while True:
		try:
			if (message == ""):
				value = int(input(bcolors.YELLOW + "Enter a value: " + bcolors.ENDC))
			else:
				value = int(input(message))
			if (value < 0):
				raise NegativeValueException
		except KeyboardInterrupt:
			print(bcolors.RED + "\nReading KeyboardInterrupt, now exiting program" + bcolors.ENDC)
			return -1
		except NegativeValueException:
			print(bcolors.RED + "I read a negative value, please try again" + bcolors.ENDC)
		except:
			print(bcolors.RED + "Could not read your value, please try again" + bcolors.ENDC)
		else:
			return value

def parse_input_float(message=""):
	value = 0
	while True:
		try:
			if (message == ""):
				value = float(input(bcolors.YELLOW + "Enter a value: " + bcolors.ENDC))
			else:
				value = float(input(message))
			if (value < 0):
				raise NegativeValueException
		except KeyboardInterrupt:
			print(bcolors.RED + "\nReading KeyboardInterrupt, now exiting program" + bcolors.ENDC)
			return -1
		except NegativeValueException:
			print(bcolors.RED + "I read a negative value, please try again" + bcolors.ENDC)
		except:
			print(bcolors.RED + "Could not read your value, please try again" + bcolors.ENDC)
		else:
			return value

def parse_csv(filename, name_value1, name_value2):
	value1, value2 = [], []
	try:
		with open(filename, 'r') as fobj:
			for row in csv.DictReader(fobj):
				value1.append(int(row[name_value1]))
				value2.append(int(row[name_value2]))
		if (len(value1) != len(value2)):
			print(bcolors.RED + "Input arrays doesn't have the same size, now exiting program" + bcolors.ENDC)
			return None, None
	except Exception as e:
		print(bcolors.RED + "Error while parsing csv file: [", e, "], now exiting program" + bcolors.ENDC) 
		return None, None
	return value1, value2

def calculate_mean(values1, values2):
	mean = [int(np.mean(values2))] * len(values1)
	return (mean)

def save_thetas(theta0, theta1):
	with open("learning_results.pkl", 'wb') as fobj:
 	       pickle.dump([theta0, theta1], fobj)

def retrieve_thetas():
	thetas = []
	filename = "learning_results.pkl"
	try:
		with open(filename, 'rb') as fobj:
			thetas = pickle.load(fobj)
		return (thetas[0], thetas[1])
	except IOError:
		return (0, 0)

def normalize(value, max_value, min_value):
	return (value / (max_value - min_value))
	
def de_normalize(value, max_value, min_value):
	return (value * (max_value - min_value))
