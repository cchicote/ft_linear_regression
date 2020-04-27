import pickle
import numpy as np
import math
import csv
import utils
import train_model
import estimate_price as est
from pprint import pprint
import plotly
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objects as go
import pandas as pd

def generate_fig(data, slope, max_iterations):
	fig_dict = {
		"data": [],
		"layout": {},
		"frames": []
	}
	fig_dict["layout"]["xaxis"] = {"range": [min(data.tab_x) + (min(data.tab_x)) * 0.1, max(data.tab_x) + (max(data.tab_x)) * 0.1], "title": "Kilometers"}
	fig_dict["layout"]["yaxis"] = {"title": "Price", "type": "log"}
	fig_dict["layout"]["hovermode"] = "closest"
	fig_dict["layout"]["sliders"] = {
		"args": [
			"transition", {
				"duration": 300,
				"easing": "cubic-in-out"
			}
		],
		"initialValue": "0",
		"plotlycommand": "animate",
		"values": max_iterations,
		"visible": True
	}
	fig_dict["layout"]["updatemenus"] = [
		{
			"buttons": [
				{
					"args": [None, {"frame": {"duration": 100, "redraw": False},
							"fromcurrent": True, "transition": {"duration": 300,
												"easing": "quadratic-in-out"}}],
					"label": "Play",
					"method": "animate"
				},
				{
					"args": [[None], {"frame": {"duration": 0, "redraw": False},
								"mode": "immediate",
								"transition": {"duration": 0}}],
					"label": "Pause",
					"method": "animate"
				}
			],
			"direction": "left",
			"pad": {"r": 10, "t": 87},
			"showactive": False,
			"type": "buttons",
			"x": 0.1,
			"xanchor": "right",
			"y": 0,
			"yanchor": "top"
		}
	]
	return fig_dict

def generate_sliders():
	sliders_dict = {
    		"active": 0,
   		"yanchor": "top",
  		"xanchor": "left",
 		"currentvalue": {
			"font": {"size": 20},
			"prefix": "Iterations:",
			"visible": True,
			"xanchor": "right"
		},
		"transition": {"duration": 300, "easing": "cubic-in-out"},
		"pad": {"b": 10, "t": 50},
		"len": 0.9,
		"x": 0.1,
		"y": 0,
		"steps": []
	}
	return sliders_dict

def calculate_sse(data, estimations):
	residuals = []
	squared_residuals = []
	for i in range(len(data.tab_y)):
		residual = estimations[i] - data.tab_y[i]
		residuals.append(residual)
		squared_residuals.append(math.pow(residual, 2))
	sse = sum(squared_residuals)
	return (sse / len(data.tab_y))

def build_fig_data(data, estimations, graph_data):
	# Generate Figure data
	data_dict_given = {
		"x": data.tab_x,
		"y": data.tab_y,
		"mode": "markers",
		"text": "Given data",
		"name": "given data"
	}
	data_dict_estimations = {
		"x": data.tab_x,
		"y": estimations,
		"mode": "markers",
		"text": "Estimations",
		"name": "estimations"
	}
	graph_data["figure"]["data"].append(data_dict_given)
	graph_data["figure"]["data"].append(data_dict_estimations)

def build_frame_data(data, estimations, i, graph_data):
	# Build frame for the given dataset
	frame = {"data": [], "name": str(i)}
	data_dict_given = {
		"x": data.tab_x,
		"y": data.tab_y,
		"mode": "markers",
		"text": "Given data",
		"name": "givendata"
	}
	frame["data"].append(data_dict_given)
	graph_data["figure"]["frames"].append(frame)
	slider_step = {"args": [
		[i],
		{"frame": {"duration": 300, "redraw": False},
		 "mode": "immediate",
		 "transition": {"duration": 300}}
	],
		"label": i,
		"method": "animate"}
	graph_data["sliders"]["steps"].append(slider_step)

	# Build frame for our estimations
	data_dict = {
		"x": data.tab_x,
		"y": estimations,
		"mode": "markers",
		"text": "Estimations",
		"name": "estimations"
		}
	frame["data"].append(data_dict)
	graph_data["figure"]["frames"].append(frame)
	slider_step = {"args": [
		[i],
		{"frame": {"duration": 300, "redraw": False},
		 "mode": "immediate",
		 "transition": {"duration": 300}}
	],
		"label": i,
		"method": "animate"}
	graph_data["sliders"]["steps"].append(slider_step)


def run_test(data, slope, max_iterations, ratio, graph_data):

	# Build initial figure data, and frame_data for i == 0
	print("Building initial figure and frame datas...", end=' ')
	estimations = []
	for x in data.tab_x:
		estimations.append(est.estimate_price(x, data.theta0, data.theta1))
	build_fig_data(data, estimations, graph_data)
	build_frame_data(data, estimations, 0, graph_data)
	print("done!")
		
	# Iterate max_iterations times through the dataset to train the model
	print("Building frame data...", end=' ')
	for i in range(1, max_iterations + 1):
		data.norm_theta0, data.norm_theta1 = train_model.train_model(ratio, data.norm_theta0, data.norm_theta1, data.norm_tab_x, data.norm_tab_y)
		# Every slope times, we make a snapshot of the estimations with the current thetas (slope can be equal to 1)
		if (i % slope == 0):
			# De-normalize our thetas to use them for the estimations
			data.theta0 = utils.de_normalize(data.norm_theta0, data.max_y, data.min_y)
			data.theta1 = utils.de_normalize(data.norm_theta1, data.max_y, data.min_y) / (data.max_x - data.min_x)
			estimations = []
			# Iterate through the dataset to estimate for each entry
			for x in data.tab_x:
				estimations.append(est.estimate_price(x, data.theta0, data.theta1))
			build_frame_data(data, estimations, i, graph_data)
	print("done!")
	print("Generating figure...", end=' ')
	graph_data["figure"]["layout"]["sliders"] = [graph_data["sliders"]]
	fig = go.Figure(graph_data["figure"])
	fig.show()
	print("done!")


def main():
	tmp_tab_x, tmp_tab_y = utils.parse_csv('data.csv', 'km', 'price')
	if (tmp_tab_x is None or tmp_tab_y is None):
		return
	data = train_model.Dataset(tmp_tab_x, tmp_tab_y)
	
	# The slope defines how many iterations the program has to go through before adding a new frame to the graph
	slope = 1
	# Does this need any explanation ?
	max_iterations = 200
	# Learning ratio
	ratio = 1
	
	graph_data = {}
	graph_data["figure"] = generate_fig(data, slope, max_iterations)
	graph_data["sliders"] = generate_sliders()
	
	run_test(data, slope, max_iterations, ratio, graph_data)

if __name__ == '__main__':
	main()
