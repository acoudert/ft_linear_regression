from sys import exit, stderr
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileinput
from estimate import estimatePrice
import config

class Trainer:

    CONFIG_FILE = "config.py"

    def __init__(self):
        with open(config.csv_file, "r") as f:
            csv.Sniffer().sniff(f.read(2048))
        df = pd.read_csv(config.csv_file)
        self.kms = np.array(df["km"]) / 1000
        self.prices = np.array(df["price"]) / 1000

    def train(self):
        theta0 = config.theta0
        theta1 = config.theta1
        kms = self.kms
        prices = self.prices
        m = kms.size
        learning_rate = config.learning_rate
        delta = config.delta_stop_regression
        divergence_limit = config.divergence_limit
        display_iteration_info = config.display_iteration_info
        display_thetas_variations = config.display_thetas_variations
        if config.display_thetas_variations:
            tmp_thetas0 = []
            tmp_thetas1 = []
        estimateFunc = np.vectorize(lambda km: estimatePrice(km, theta0, theta1))
        i = 0
        while True:
            temp = estimateFunc(kms) - prices
            tmp_theta0 = learning_rate * sum(temp) / m
            tmp_theta1 = learning_rate * sum(temp * kms) / m
            theta0 -= tmp_theta0
            theta1 -= tmp_theta1
            if display_thetas_variations:
                tmp_thetas0.append(tmp_theta0)
                tmp_thetas1.append(tmp_theta1)
            if abs(tmp_theta0) < delta and abs(tmp_theta1) < delta:
                self.theta0 = 1000 * theta0
                self.theta1 = theta1
                if display_thetas_variations:
                    self.displayThetasVariations(tmp_thetas0, tmp_thetas1)
                return
            if display_iteration_info:
                i += 1
                print(i, theta0, theta1, tmp_theta0, tmp_theta1)
            if abs(theta0) > divergence_limit or abs(theta1) > divergence_limit:
                print("Function is divergent - ABORT", file=stderr)
                exit(1)

    def save(self):
        for l in fileinput.input(self.CONFIG_FILE, inplace=True):
            if l.find("theta0") != -1:
                print("theta0 =", self.theta0)
            elif l.find("theta1") != -1:
                print("theta1 =", self.theta1)
            else:
                print(l, end="")
    
    def displayThetasVariations(self, tmp_thetas0, tmp_thetas1):
        plt.figure(figsize=(10, 9))
        plt.subplot(211)
        plt.plot(range(0, len(tmp_thetas0)), tmp_thetas0, "g-", 
                label="tmp_theta0")
        plt.xlabel("iteration")
        plt.ylabel("tmp_theta0")
        plt.legend()
        plt.subplot(212)
        plt.plot(range(0, len(tmp_thetas1)), tmp_thetas1, "b-", 
                label="tmp_theta1")
        plt.xlabel("iteration")
        plt.ylabel("tmp_theta1")
        plt.legend()
        plt.show()

    def displayEstimatedVsData(self):
        if not hasattr(self, "theta0") or not hasattr(self, "theta1"):
            self.theta0 = config.theta0
            self.theta1 = config.theta1
        plt.figure(figsize=(10, 7))
        plt.plot(self.kms * 1000, self.prices * 1000, "go", 
                label="data.csv", markersize=3)
        x = np.arange(10000, 245001, 5000)
        f = np.vectorize(lambda km: estimatePrice(km, self.theta0, self.theta1))
        y = f(x)
        plt.plot(x, y, "bo-", label="estimated", markersize=3)
        plt.axis([5000, 250000, 3000, 9000])
        plt.xlabel("km")
        plt.ylabel("price")
        plt.title("Estimated vs Data")
        plt.legend()
        plt.show()

def main():
    try:
        t = Trainer()
        t.train()
        t.save()
        if config.display_estimated_vs_datas:
            t.displayEstimatedVsData()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
