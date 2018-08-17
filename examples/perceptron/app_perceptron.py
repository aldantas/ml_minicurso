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
from gi import require_version
require_version('Gtk', '3.0')
import matplotlib
matplotlib.use('GTK3Agg')
from gi.repository import Gtk
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from perceptron import Perceptron
from dataset_generator import DatasetGenerator
from data_plotter import DataPlotter
from threading import Thread


class AppPerceptron:

    def __init__(self):
        builder = Gtk.Builder()
        builder.add_from_file('main_gui.glade')
        builder.connect_signals(self)

        self.generator = DatasetGenerator()
        self.perceptron = Perceptron()
        self.train_plotter = DataPlotter()
        self.train_canvas = FigureCanvas(self.train_plotter.fig)

        self.dataset_size_entry = builder.get_object('dataset_size_entry')
        self.max_epochs_entry = builder.get_object('max_epochs_entry')
        self.test_size_entry = builder.get_object('test_r_size_entry')
        self.learning_rate_entry = builder.get_object('learning_r_entry')
        self.epoch_output = builder.get_object('epoch_output')
        self.train_errors_output = builder.get_object('train_errors_output')
        self.test_errors_output = builder.get_object('test_errors_output')
        self.model_output = builder.get_object('model_output')

        self.gen_data_button = builder.get_object('gen_data_button')
        self.train_button = builder.get_object('train_button')
        self.test_button = builder.get_object('test_button')

        box = builder.get_object('graph_box')
        box.pack_start(self.train_canvas, True, True, 0)
        self.on_gen_data_button_clicked()

        window = builder.get_object('main_window')
        window.set_default_size(700, 550)
        window.show_all()

    def on_delete_window(self, *args):
        Gtk.main_quit(*args)

    def on_gen_data_button_clicked(self, widget=None):
        self.clear_status_bar()
        self.test_button.set_sensitive(False)
        dataset_size = int(self.dataset_size_entry.get_text())
        test_rate_size = self.test_size_entry.get_value()
        split_data, self.original_model = self.generator.get_split_dataset(
            dataset_size, test_rate_size)
        self.train_set, self.test_set = split_data
        self.base_plot()

    def on_train_button_clicked(self, widget):
        self.disable_buttons()
        self.clear_status_bar()
        self.base_plot()
        max_epochs = int(self.max_epochs_entry.get_text())
        learning_rate = self.learning_rate_entry.get_value()
        train_method = self.perceptron.train
        thread = Thread(target=train_method, args=(self.train_set,
                                                   self.train_plotter,
                                                   self.on_train_finished,
                                                   max_epochs, learning_rate))
        thread.daemon = True
        thread.start()

    def on_test_button_clicked(self, widget):
        testing_results, test_errors = self.perceptron.test(self.test_set)
        self.train_plotter.plot_dataset(testing_results, True)
        model = self.perceptron.best_weights
        self.train_plotter.plot_decision_boundary(model, color='k',
                                                  label='Trained Model')
        self.test_errors_output.set_text(str(test_errors))

    def on_train_finished(self, trainning_results):
        self.update_status_bar(trainning_results)
        self.enable_buttons()

    def base_plot(self):
        self.perceptron.clear_line()
        self.train_plotter.plot_dataset(self.train_set)
        model = self.original_model
        self.train_plotter.plot_decision_boundary(model, color='g',
                                                  label='Original Model')

    def enable_buttons(self):
        self.gen_data_button.set_sensitive(True)
        self.train_button.set_sensitive(True)
        self.test_button.set_sensitive(True)

    def disable_buttons(self):
        self.gen_data_button.set_sensitive(False)
        self.train_button.set_sensitive(False)
        self.test_button.set_sensitive(False)

    def clear_status_bar(self):
        self.epoch_output.set_text('--')
        self.train_errors_output.set_text('--')
        self.test_errors_output.set_text('--')
        self.model_output.set_text('--')

    def update_status_bar(self, results):
        model, epoch, best_error = results
        self.epoch_output.set_text(str(epoch))
        self.train_errors_output.set_text(str(best_error))
        self.test_errors_output.set_text('--')
        signs = []
        for weight in model:
            signs.append('+' if weight > 0 else '-')
        self.model_output.set_text(
            '{0:.4f} x1 {1} {2:.4f} x2 {3} {4:.4f}'.format(
                model[1], signs[2], abs(model[2]), signs[0], abs(model[0])))


if __name__ == '__main__':
    view = AppPerceptron()
    Gtk.main()
