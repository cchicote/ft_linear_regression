import numpy as np
import utils
import estimate_price as est

class Dataset:
    def __init__(self, tab_x, tab_y):
        self.tab_x = np.array(tab_x, dtype=np.float64)
        self.tab_y = np.array(tab_y, dtype=np.float64)
        self.min_x = self.tab_x.min()
        self.max_x = self.tab_x.max()
        self.min_y = self.tab_y.min()
        self.max_y = self.tab_y.max()
        self.norm_tab_x = utils.normalize(self.tab_x, self.max_x, self.min_x)
        self.norm_tab_y = utils.normalize(self.tab_y, self.max_y, self.min_y)
        self.sorted_tab_x, self.sorted_tab_y = zip(*sorted(zip(self.tab_x, self.tab_y)))
        self.theta0 = 0
        self.norm_theta0 = 0
        self.theta1 = 0
        self.norm_theta1 = 0
        self.mean_x = utils.calculate_mean(self.tab_y, self.tab_x)
        self.mean_y = utils.calculate_mean(self.tab_x, self.tab_y)

def train_model(ratio, theta0, theta1, tab_x, tab_y):
    m = len(tab_x)
    tmp_t0, tmp_t1, estimated_price = 0, 0, 0
    for i in range(m):
        estimated_price = est.estimate_price(tab_x[i], theta0, theta1)
        tmp_t0 += ratio * (estimated_price - tab_y[i])
        tmp_t1 += ratio * (estimated_price - tab_y[i]) * tab_x[i]
    theta0 -= tmp_t0 / m
    theta1 -= tmp_t1 / m
    return (theta0, theta1)

def main():
    # Get dataset - Get min-max values - Normalize values - Sort tabs - Set both thetas to 0
    tmp_tab_x, tmp_tab_y = utils.parse_csv('data.csv', 'km', 'price')
    if (tmp_tab_x is None or tmp_tab_y is None):
        return
    data = Dataset(tmp_tab_x, tmp_tab_y)

    # Get number of trainings and ratio
    loops = utils.parse_input_int(message=utils.bcolors.YELLOW + "How much trainings do you want our program to go through:\n" + utils.bcolors.ENDC)
    if loops == -1:
        return
    ratio = utils.parse_input_float(message=utils.bcolors.YELLOW + "At what rate:\n" + utils.bcolors.ENDC)
    if ratio == -1:
        return

    # train model
    for _ in range(loops):
        data.norm_theta0, data.norm_theta1 = train_model(ratio, data.norm_theta0, data.norm_theta1, data.norm_tab_x, data.norm_tab_y)

    # de-normalize thetas and save them
    data.theta0 = utils.de_normalize(data.norm_theta0, data.max_y, data.min_y)
    data.theta1 = utils.de_normalize(data.norm_theta1, data.max_y, data.min_y) / (data.max_x - data.min_x)
    utils.save_thetas(data.theta0, data.theta1)

if __name__ == '__main__':
    main()
