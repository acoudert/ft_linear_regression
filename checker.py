from sys import exit, stderr
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileinput
from estimate import estimatePrice
import config

class Checker:

    CONFIG_FILE = "config.py"

    def __init__(self):
        with open(config.csv_file, "r") as f:
            csv.Sniffer().sniff(f.read(2048))
        df = pd.read_csv(config.csv_file)
        self.kms = np.array(df["km"])
        self.prices = np.array(df["price"])

    def bestLineDescription(self):
        kms = self.kms
        prices = self.prices
        average_x = sum(kms) / kms.size
        average_y = sum(prices) / prices.size
        numer = 0
        denom = 0
        for i in range(kms.size):
            numer += (kms[i] - average_x) * (prices[i] - average_y)
            denom += (kms[i] - average_x) ** 2
        self.slope = numer / denom
        self.y_intercept = average_y - self.slope * average_x

    def display(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.kms, self.prices, "go", 
                label="data.csv", markersize=3)
        self.x = np.arange(10000, 245001, 5000)
        f = np.vectorize(lambda km: estimatePrice(km, self.y_intercept, self.slope))
        self.best_line_description = f(self.x)
        plt.plot(self.x, self.best_line_description, "r-", 
                label="best line description", markersize=3)
        f = np.vectorize(lambda km: estimatePrice(km, config.theta0, config.theta1))
        self.estimated = f(self.x)
        plt.plot(self.x, self.estimated, "b-", 
                label="estimated", markersize=3)
        if config.theta0 == 0:
            plt.axis([5000, 250000, 0, 9000])
        else:
            plt.axis([5000, 250000, 3000, 9000])
        plt.xlabel("km")
        plt.ylabel("price")
        plt.title("Data")
        plt.legend()
        plt.show()

    def findAccuracy(self):
        self.percentages = np.zeros(self.x.size)
        for i in range(self.x.size):
            a = self.estimated[i]
            b = self.best_line_description[i]
            self.percentages[i] = (a / b) * 100 
        average_accuracy = sum(self.percentages) / self.percentages.size
        if average_accuracy > 100:
            average_accuracy = 100 - (average_accuracy - 100)
        print("Accuracy: {:.3f}%".format(average_accuracy))

def main():
    try:
        t = Checker()
        t.bestLineDescription()
        t.display()
        t.findAccuracy()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
