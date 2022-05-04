from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math


class PerformanceMeasure:

    def __init__(self, actual_values, predictions):
        self.actual_values = actual_values
        self.predictions = predictions

    def metrics(self):
        r2 = r2_score(self.actual_values, self.predictions)
        mae = mean_absolute_error(self.actual_values, self.predictions)
        mse = mean_squared_error(self.actual_values, self.predictions)
        rmse = math.sqrt(mse)

        print("R Squared: ", r2)
        print("Mean Absolute Error: ", mae)
        print("Root Mean Square Error: ", rmse)
        print("Mean Squared Error: ", mse)

    def plot(self):
        plt.plot(self.actual_values, color='#34a56f')
        plt.plot(self.predictions, color='#ff0000')  # 1589ff
        plt.legend(['Actual', 'Predicted'])
        plt.title("Energy use prediction using GB model")
        plt.xlabel("Instances")
        plt.ylabel("Energy Value")
        plt.show()
