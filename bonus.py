import numpy as np
import plotly.graph_objects as go
import utils
import train_model
import estimate_price as est


def generate_fig(max_iterations, title_x, title_y):
    # Generate the figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    fig_dict["layout"]["xaxis"] = {"title": title_x}
    fig_dict["layout"]["yaxis"] = {"title": title_y}
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
    # Generate the sliders
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

def build_fig_data_linear_regression(data, estimations, graph_data):
    # Generate the figure's data
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
        "mode": "lines+markers",
        "text": "Estimations",
        "name": "estimations"
    }
    graph_data["figure"]["data"].append(data_dict_given)
    graph_data["figure"]["data"].append(data_dict_estimations)

def build_frame_data_linear_regression(data, estimations, i, graph_data):
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
        "mode": "lines+markers",
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

def plot_linear_regression(data, slope, max_iterations, ratio):
    utils.print_colored("[linear_regression][Plotting linear regression...] start!", utils.bcolors.YELLOW)
    graph_data = {}
    graph_data["figure"] = generate_fig(max_iterations, "Kilometers", "Price")
    graph_data["sliders"] = generate_sliders()

    # Build initial figure data, and frame_data for i == 0
    utils.print_colored("[linear_regression][Building initial figure and frame datas...] start!", utils.bcolors.YELLOW)
    estimations = []
    for x in data.tab_x:
        estimations.append(est.estimate_price(x, data.theta0, data.theta1))
    build_fig_data_linear_regression(data, estimations, graph_data)
    build_frame_data_linear_regression(data, estimations, 0, graph_data)
    utils.print_colored("[linear_regression][Building initial figure and frame datas...] finished!", utils.bcolors.GREEN)

    # Iterate max_iterations times through the dataset to train the model
    utils.print_colored("[linear_regression][Building frame data...] start!", utils.bcolors.YELLOW)
    for i in range(1, max_iterations + 1):
        data.norm_theta0, data.norm_theta1 = train_model.train_model(ratio, data.norm_theta0, data.norm_theta1, data.norm_tab_x, data.norm_tab_y)
        # Every slope times, we make a snapshot of the estimations with the current thetas (slope can be equal to 1)
        if i % slope == 0:
            # De-normalize our thetas to use them for the estimations
            data.theta0 = utils.de_normalize(data.norm_theta0, data.max_y, data.min_y)
            data.theta1 = utils.de_normalize(data.norm_theta1, data.max_y, data.min_y) / (data.max_x - data.min_x)
            estimations = []
            # Iterate through the dataset to estimate for each entry
            for x in data.tab_x:
                estimations.append(est.estimate_price(x, data.theta0, data.theta1))
            build_frame_data_linear_regression(data, estimations, i, graph_data)
    utils.print_colored("[linear_regression][Building frame data...] finished!", utils.bcolors.GREEN)

    utils.print_colored("[linear_regression][Generating figure...] start!", utils.bcolors.YELLOW)
    graph_data["figure"]["layout"]["sliders"] = [graph_data["sliders"]]
    fig = go.Figure(graph_data["figure"])
    fig.show()
    utils.print_colored("[linear_regression][Generating figure...] finished!", utils.bcolors.GREEN)
    utils.print_colored("[linear_regression][Plotting linear regression...] finished!\n", utils.bcolors.GREEN)

def cost_function(data):
    cost = 0
    m = len(data.norm_tab_x)
    for i in range(len(data.norm_tab_x)):
        cost += (est.estimate_price(data.norm_tab_x[i], data.norm_theta0, data.norm_theta1) - data.norm_tab_y[i])**2
    return cost/(2*m)

def plot_cost_function(data, max_iterations, ratio):
    utils.print_colored("[cost_function][Plotting cost function...] start!", utils.bcolors.YELLOW)
    cost = []

    utils.print_colored("[cost_function][Calculating cost function...] start!", utils.bcolors.YELLOW)
    for _ in range(1, max_iterations + 1):
        data.norm_theta0, data.norm_theta1 = train_model.train_model(ratio, data.norm_theta0, data.norm_theta1, data.norm_tab_x, data.norm_tab_y)
        cost.append(cost_function(data))
    utils.print_colored("[cost_function][Calculating cost function...] finished!", utils.bcolors.GREEN)

    utils.print_colored("[cost_function][Generating figure...] start!", utils.bcolors.YELLOW)
    max_iter_table = np.arange(max_iterations)
    fig = go.Figure(data=go.Scatter(x=max_iter_table, y=cost))
    fig.show()
    utils.print_colored("[cost_function][Generating figure...] finished!", utils.bcolors.GREEN)
    utils.print_colored("[cost_function][Plotting cost function...] finished!\n", utils.bcolors.GREEN)

def main():
    # The slope defines how many iterations the program has to go through before adding a new frame to the graph
    slope = 1
    # Does this need any explanation ?
    max_iterations = 1000
    # Learning ratio
    ratio = 0.1

    # Parsing CSV to build the data
    tmp_tab_x, tmp_tab_y = utils.parse_csv('data.csv', 'km', 'price')
    if (tmp_tab_x is None or tmp_tab_y is None):
        return

    # Creating new dataset for cost function test
    data = train_model.Dataset(tmp_tab_x, tmp_tab_y)
    plot_cost_function(data, max_iterations, ratio)

    # Creating new dataset for linear regression test
    data = train_model.Dataset(tmp_tab_x, tmp_tab_y)
    plot_linear_regression(data, slope, max_iterations, ratio)

if __name__ == '__main__':
    main()
