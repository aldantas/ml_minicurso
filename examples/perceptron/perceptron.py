# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from gi.repository import GLib
import numpy as np


class Perceptron:

    def __init__(self):
        self.line = None

    def clear_line(self):
        self.line = None

    def train(self, train_set, plotter, callback, max_epochs=500,
              learning_rate=0.7):
        weights = np.array([np.random.uniform(-1, 1) for _ in range(3)])
        best_error = float('inf')
        for epoch in range(1, max_epochs+1):
            epoch_error = 0
            for entry in train_set:
                values, label = entry[0], entry[1]
                prediction = 1 if weights.dot(values) >= 0 else 0
                error = label - prediction
                if error:
                    weights += error * values * learning_rate
                    epoch_error += 1
            self.line = plotter.plot_decision_boundary(weights, self.line,
                                                       color='k',
                                                       label='Trainning Model')
            if epoch_error < best_error:
                best_weights = weights
                best_epoch = epoch
                best_error = epoch_error
                if epoch_error == 0:
                    break
        else:
            self.line = plotter.plot_decision_boundary(best_weights, self.line,
                                                       color='k',
                                                       label='Trainning Model')
        self.best_weights = best_weights
        results =  best_weights, best_epoch, best_error
        GLib.idle_add(callback, results)

    def test(self, test_set):
        try:
            prediction_list = []
            test_errors = len(test_set)
            for entry in test_set:
                values, label = entry[0], entry[1]
                prediction = 1 if self.best_weights.dot(values) >= 0 else 0
                is_prediction_right = True if prediction == label else False
                test_errors -= is_prediction_right
                prediction_list.append((entry[0], entry[1],
                                        is_prediction_right))
            return prediction_list, test_errors
        except AttributeError:
            return [], 0
