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
import numpy as np


class DatasetGenerator:

    def generate_dataset(self, size):
        self.set_random_target_line()
        self.dataset = []
        for i in range(size):
            x1, x2 = [np.random.uniform(-1, 1) for _ in range(2)]
            attrs = np.array([1, x1, x2])
            label = 1 if self.target_line.dot(attrs) >= 0 else 0
            self.dataset.append((attrs, label))

    def split_dataset(self, test_proportion=.3):
        dataset = np.array(self.dataset)
        test_set, train_set = [], []
        for label in (0, 1):
            label_set = dataset[dataset[:, 1] == label].tolist()
            test_size = round(test_proportion * len(label_set))
            test_set.extend(label_set[:test_size])
            train_set.extend(label_set[test_size:])
        np.random.shuffle(test_set)
        np.random.shuffle(train_set)
        return train_set, test_set

    def set_random_target_line(self):
        w1, w2 = [np.random.uniform(-1, 1) for _ in range(2)]
        # make bias close to 0 to force a fair class division
        b = np.random.uniform(-.15, .15)
        self.target_line = np.array([b, w1, w2])

    def get_split_dataset(self, size, test_proportion):
        self.generate_dataset(size)
        return self.split_dataset(test_proportion), self.target_line
