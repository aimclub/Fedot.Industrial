import numpy as np
from matplotlib import pyplot as plt


class DataProjection:
    """
    A class to project data into a different basis.

    Attributes:
        data (np.array): An array of data to be projected.
    """

    def __init__(self, data):
        """
        The constructor for the DataProjection class.

        Parameters:
            data (np.array): An array of data to be projected.
        """
        self.data = data

    def project_data(self, basis):
        """
        Projects the data into a different basis.

        Parameters:
            basis (np.array): The basis to project the data into.

        Returns:
            np.array: The projected data.
        """
        return np.dot(self.data, basis)

    def visualize_data(self):
        """
        Visualizes the data.
        """
        plt.scatter(self.data[:, 0], self.data[:, 1])
        plt.show()

    def visualize_projection(self, basis):
        """
        Visualizes the data projected into a different basis.

        Parameters:
            basis (np.array): The basis to project the data into.
        """
        projected_data = self.project_data(basis)
        plt.scatter(projected_data[:, 0], projected_data[:, 1])
        plt.show()

    def __add__(self, other):
        """
        Adds two DataProjection classes together.

        Parameters:
            other (DataProjection): The other DataProjection class to add.

        Returns:
            DataProjection: The combined DataProjection classes.
        """
        return DataProjection(self.data + other.data)

    def __sub__(self, other):
        """
        Subtracts two DataProjection classes.

        Parameters:
            other (DataProjection): The other DataProjection class to subtract.

        Returns:
            DataProjection: The subtracted DataProjection classes.
        """
        return DataProjection(self.data - other.data)

    def __mul__(self, other):
        """
        Multiplies two DataProjection classes.

        Parameters:
            other (DataProjection): The other DataProjection class to multiply.

        Returns:
            DataProjection: The multiplied DataProjection classes.
        """
        return DataProjection(self.data * other.data)
