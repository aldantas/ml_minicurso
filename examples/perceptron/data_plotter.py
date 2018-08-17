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
import matplotlib.pyplot as plt
import numpy as np
import time


class DataPlotter:

    def __init__(self):
        self.fig = plt.figure()
        self.configure_axis()

    def configure_axis(self):
        self.fig.gca().axis([-1, 1, -1, 1])
        self.fig.gca().set_xlabel('x1')
        self.fig.gca().set_ylabel('x2')

    def plot_decision_boundary(self, weights, line=None, label=None,
                               color=None):
        xx = [-1, 1]
        yy = (-np.dot(weights[1], xx) - weights[0]) / weights[2]
        if not line:
            line, = self.fig.gca().plot(xx, yy, color=color, linewidth=.5,
                                        label=label)
            self.fig.gca().legend(loc='lower center',
                                  bbox_to_anchor=(0., 1.02, 1., .102),
                                  fancybox=True, shadow=True, ncol=2,
                                  borderaxespad=0)
            GLib.idle_add(self.fig.canvas.draw_idle)
        else:
            line.set_ydata(yy)
            GLib.idle_add(self.fig.canvas.draw_idle)
            time.sleep(0.1)
        return line

    def plot_dataset(self, dataset, is_testing=False):
        if len(dataset) == 0:
            return
        self.fig.clf()
        self.configure_axis()
        for data in dataset:
            marker = 'o' if data[1] else 'x'
            if is_testing:
                color = 'g' if data[2] else 'r'
            else:
                color = 'b'
            self.fig.gca().plot(data[0][1], data[0][2],
                                marker, color=color, markersize=4)
        self.fig.canvas.draw_idle()
