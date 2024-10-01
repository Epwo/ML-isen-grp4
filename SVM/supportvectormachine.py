import pandas as pd


class SupportVectorMachine:

    def __init__(self, df):
        self.df = df
        self.population = None
        self.price = None

    def extractData(self):
        self.population = self.df["population"]
        self.price = self.df["Price"]

    def predict(self):
        pass

    def fit(self):
        pass
