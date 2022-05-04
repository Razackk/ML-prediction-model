import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn import tree
from sklearn.svm import SVR


class RegressionModels:

    def __init__(self, file_path):
        self.file_path = file_path

    def data_split(self):
        df = pd.read_csv(self.file_path)

        x = df.drop(['ENERGY_CONSUMPTION'], axis=1)
        y = df['ENERGY_CONSUMPTION_CURRENT']

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

    def neural_network(self):
        x_train, x_test, y_train, y_test = self.data_split()

        reg = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)
        predictions = reg.predict(x_test)

        return y_test, predictions

    def support_vector(self):
        x_train, x_test, y_train, y_test = self.data_split()

        svr = SVR()
        reg = svr.fit(x_train, y_train)

        predictions = reg.predict(x_test)

        return y_test, predictions

    def decision_tree(self):
        x_train, x_test, y_train, y_test = self.data_split()
        clf = tree.DecisionTreeRegressor()

        clf = clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)

        return y_test, predictions

    def gradient_boosting(self):
        x_train, x_test, y_train, y_test = self.data_split()

        est = GradientBoostingRegressor().fit(x_train, y_train)
        predictions = est.predict(x_test)

        return y_test, predictions

    def nearest_neighbor(self):
        x_train, x_test, y_train, y_test = self.data_split()

        knn = KNeighborsRegressor()
        reg = knn.fit(x_train, y_train)

        predictions = reg.predict(x_test)

        return y_test, predictions

    def deep_neural_network(self):
        x_train, x_test, y_train, y_test = self.data_split()

        reg = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="relu", random_state=1, max_iter=2000).\
            fit(x_train, y_train)

        predictions = reg.predict(x_test)

        return y_test, predictions

    def random_forest(self):
        x_train, x_test, y_train, y_test = self.data_split()
        reg = RandomForestRegressor()
        reg.fit(x_train,y_train)

        predictions = reg.predict(x_test)

        return y_test, predictions

    def stacking(self):
        x_train, x_test, y_train, y_test = self.data_split()
        estimators = [('lr', RidgeCV()), ('svr', LinearSVR(random_state=42))]
        reg = StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor(n_estimators=10,
                                                  random_state=42)
         )
        reg.fit(x_train, y_train)
        predictions = reg.predict(x_test)

        return y_test, predictions

    def linear_regression(self):
        x_train, x_test, y_train, y_test = self.data_split()
        lm = LinearRegression()
        lm.fit(x_train, y_train)

        predictions = lm.predict(x_test)

        return y_test, predictions
