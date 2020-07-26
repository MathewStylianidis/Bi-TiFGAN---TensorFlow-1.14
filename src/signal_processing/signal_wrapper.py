"""
 This file includes the Signal class definition which is a wrapper around a one dimensional
 numpy array object providing functionality around signal processing and plotting.
"""

__author__ = "Matthaios Stylianidis"

import numpy as np
import matplotlib.pyplot as plt


class SignalWrapper:
    """ Class for manipulating signals.

    Attributes:
        signal: A numpy array with the signal
        signal_len: The signal's length
    """

    def __init__(self, signal):
        self.signal = signal
        self.signal_len = len(self.signal)

    def meets_amplitude_requirement(self, constant_factor):
        """ Checks if the signal meets the amplitude requirement.

        The amplitude requirement is defined by <factor> * len(signal).

        Args:
            constant_factor: The constant factor multiplied by the length of the signal
                to determine if the signal amplitude is overall high enough.

        Returns:
            True if the requirement is met. False otherwise.
        """
        amplitude_sum = np.sum(np.absolute(self.signal))
        minimum_amplitude_requirement = self.signal_len * constant_factor
        if amplitude_sum < minimum_amplitude_requirement:
            return False
        return True

    def pad_or_crop(self, fixed_len, mode="side", in_place=False):
        """ Pads or crops the sides of a signal to meet length requirements.

        Args:
            fixed_len: The desired length of the numpy array.
            mode: The type of padding to be applied. Defaults to 'side':
                {
                    mode='side': Pads or crops the sides of the signal.
                    mode='end': Pads or crops the end of the signal.
                }
            in_place: If True the signal attribute is replaced with the cropped
                or padded signal. If False the method returns the cropped or pad-
                ded signal. Defaults to False.
        Returns:
            The cropped or padded one dimensional numpy array.
        """
        if self.signal_len > fixed_len:
            edited_signal = self.crop(fixed_len, mode=mode, in_place=False)
        else:
            edited_signal = self.pad(fixed_len, mode=mode, in_place=False)

        if in_place:
            self.signal = edited_signal
        else:
            return edited_signal

    def pad(self, fixed_len, mode, in_place=False):
        """ Pads the signal according to the specified mode.

        fixed_len: The desired length of the signal.
        mode: The type of padding to be applied. Defaults to 'side':
            {
                mode='side': Pads the sides of the signal.
                mode='end': Pads the end of the signal.
            }
        in_place: If True the signal attribute is replaced with the padded signal.
            If False the method returns the padded signal. Defaults to False.
        """
        # Check signal length
        total_pad_length = fixed_len - self.signal_len
        assert total_pad_length >= 0
        # Check correct mode
        assert mode in ["side", "end"]
        if mode == "side":
            leading_zero_num = int(np.floor(total_pad_length / 2))
        elif mode == "end":
            leading_zero_num = 0
        return self.pad_sides(total_pad_length, leading_zero_num, in_place)

    def pad_sides(self, total_pad_length, leading_zero_num, in_place=False):
        """Pads the signal from both sides given the number of the total and the leading number of zeros.

        Args:
            total_pad_length: The total number of zeros in the padding from both sides.
            leading_zero_num: The leading number of zeros in the padding from the left.
            in_place: If True the signal attribute is replaced with the padded signal.
                If False the method returns the padded signal. Defaults to False.

        Returns:
            The padded signal if in_place is True. None otherwise.
        """
        trailing_zero_num = total_pad_length - leading_zero_num
        edited_signal = np.pad(self.signal, pad_width=(leading_zero_num, trailing_zero_num), mode='constant')
        if in_place:
            self.signal = edited_signal
        else:
            return edited_signal

    def crop(self, fixed_len, mode, in_place=False):
        """ Crops the signal according to the specified mode.

        fixed_len: The desired length of the signal.
        mode: The type of cropping to be applied. Defaults to 'side':
            {
                mode='side': Crops the sides of the signal.
                mode='end': Crops the end of the signal.
            }
        in_place: If True the signal attribute is replaced with the cropped signal.
            If False the method returns the cropped signal. Defaults to False.
        """
        assert mode in ["side", "end"]
        if mode == "side":
            return self.crop_sides(fixed_len, in_place)
        elif mode == "end":
            return self.crop_end(fixed_len, in_place)

    def crop_end(self, fixed_len, in_place=False):
        """Crops the signal from its end.

        Args:
            fixed_len: The desired length of the signal
            in_place: If True the signal attribute is replaced with the cropped signal.
                If False the method returns the cropped signal. Defaults to False.

        Returns:
            The cropped signal if in_place is True. None otherwise.
        """
        assert self.signal_len < fixed_len
        if in_place:
            self.signal = self.signal[:fixed_len]
        else:
            return self.signal[:fixed_len]

    def crop_sides(self, fixed_len, in_place=False):
        """Crops the sides of the signal.

        Args:
            fixed_len: The desired length of the signal
            in_place: If True the signal attribute is replaced with the cropped signal.
                If False the method returns the cropped signal. Defaults to False.

        Returns:
            The cropped signal if in_place is True. None otherwise.
        """
        assert self.signal_len > fixed_len

        total_length_diff = self.signal_len - fixed_len
        leading_crop_size = int(np.floor(total_length_diff / 2))
        trailing_crop_size = int(np.ceil(total_length_diff / 2))

        edited_signal = self.signal[:self.signal_len - trailing_crop_size]
        edited_signal = edited_signal[leading_crop_size:]

        if in_place:
            self.signal = edited_signal
        else:
            return edited_signal

    def chunk(self, chunk_length, chunk_overlap):
        """Chunks the signal into parts.

        Chunks the signal into parts of equal length and returns a list of chunks as numpy arrays.
        The last chunk of the signal is discarded if the signal is not exactly dividable given the
        chunk size and chunk overlap.

        Args:
            chunk_length: The length of each chunk
            chunk_overlap: The overlap between adjacent chunks.

        Returns:
            A list of numpy arrays where each array represents a chunk of the signal.
        """
        assert self.signal_len > chunk_length
        assert chunk_overlap >= 0

        index = 0
        chunk_list = []
        step_length = chunk_length - chunk_overlap
        while index + chunk_length <= self.signal_len:
            chunk = self.signal[index:index + chunk_length]
            chunk_list.append(chunk)
            index += step_length

        return chunk_list

    def get_higher_energy_window(self, window_size):
        """ Gets the part of the signal with the specified size that has the highest energy level.

        Args:
            window_size: The size of the window of the signal to be extracted.

        Returns:
            The part of the signal with size <window-size> with the highest energy level.
        """

        assert window_size <= self.signal_len
        magnitude_squared = self.signal ** 2
        highest_energy = 0
        highest_energy_window = None
        for i in range(self.signal_len - window_size + 1):
            window = magnitude_squared[i:i+window_size]
            window_energy = window.sum()
            if window_energy > highest_energy:
                highest_energy = window_energy
                highest_energy_window = self.signal[i:i+window_size]
        return highest_energy_window

    def get_signal(self):
        """Signal getter"""
        return self.signal

    def plot(self, figure=1, title=None, title_size=14, x_axis=None, y_axis=None, x_label_size=12, y_label_size=12,
             add_subplot_no=(1, 1, 1), fig_size=None, **kwargs):
        """Plotting method wrapping up matplotlib.pyplot functionality.

        add_subplot_no does not work properly yet.

        Args:
            figure: Figure to be used.
            title: Figure title.
            title_size: Figure title.
            x_axis: The label string in the x-axis.
            y_axis: The label string in the y-axis.
            x_label_size: Title size of x-axis.
            y_label_size: Title size of y-axis.
            add_subplot_no: Tuple of first three parameters to be passed to fig.add_subplot. Currently not working
                properly.
            fig_size: Tuple with the dimensions of the figure.
            kwargs: Rest of the matplotlib.pyplot.plot parameters.
        """
        assert len(add_subplot_no) == 3
        fig = plt.figure(figure, figsize=fig_size)
        ax = fig.add_subplot(*add_subplot_no)
        plt.xlabel(x_axis, size=x_label_size)
        plt.ylabel(y_axis, size=y_label_size)
        ax.title.set_text(title)
        ax.title.set_size(title_size)
        ax.plot(self.signal, **kwargs)

























