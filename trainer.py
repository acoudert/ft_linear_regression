from sys import argv
import csv
import numpy as np
import pandas as pd
import matplotlib as plt
from calculator import estimatePrice
import config

class Trainer:

    def __init__(self, csv_file):
        with open(csv_file, "r") as f:
            csv.Sniffer().sniff(f.read(2048))
        self.df = pd.read_csv(csv_file)

    def train(self):
        d = self.df
        theta0 = config.theta0
        theta1 = config.theta1
        self.df["estimated"] = 0
        estimateFunc = lambda r: estimatePrice(r["km"], theta0, theta1) - r["price"]
        theta0SumFunc = lambda max_i: sum([estimateFunc(r) \
                for i, r in d.iterrows() if i < max_i])
        theta1SumFunc = lambda max_i: sum([estimateFunc(r) * r["km"] \
                for i, r in d.iterrows() if i < max_i])
        for i, r in self.df.iterrows():
            print(i, theta0, theta1, estimatePrice(61789, theta0, theta1))
            tmp_theta0 = config.learning_rate * theta0SumFunc(i+1) / (i+1)
            tmp_theta1 = config.learning_rate * theta1SumFunc(i+1) / (i+1)
            theta0 = tmp_theta0
            theta1 = tmp_theta1
            print(i, theta0, theta1, estimatePrice(61789, theta0, theta1))
            input()
            

def main():
    try:
        t = Trainer(argv[1])
        t.train()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
