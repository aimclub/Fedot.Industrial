import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TimeSeriesGenerator:

    def __init__(self, n_periods, n_ts, length: int = 30):
        self.n_periods = n_periods
        self.length = length
        self.n_ts = n_ts
        self.freq = 'D'

    def generate_sine_wave(self, amplitude, period):
        time_index = pd.date_range(start='2023-01-01', periods=self.n_periods, freq=self.freq)
        sine_wave = amplitude * np.sin(2 * np.pi / period * time_index.dayofyear)
        return sine_wave

    def generate_random_walk(self, start_val):
        # time_index = pd.date_range(start='2023-01-01', periods=self.n_periods, freq=self.freq)
        time_index = pd.date_range(start='2023-01-01', periods=self.n_periods, freq=self.freq)
        time_index = pd.Series(np.arange(0, self.length))
        random_walk = pd.Series(np.cumsum(np.random.randn(self.n_periods)) + start_val, index=time_index)
        return random_walk

    def generate_autoregression(self, ar_params):
        time_index = pd.date_range(start='2023-01-01', periods=self.n_periods, freq=self.freq)
        ar_process = pd.Series(np.zeros(self.n_periods), index=time_index)
        for i in range(len(ar_process)):
            if i == 0:
                ar_process.iloc[i] = np.random.normal(0, 1, self.n_ts)
            else:
                ar_process.iloc[i] = np.sum(ar_params * ar_process.iloc[i - len(ar_params):i, :],
                                            axis=1) + np.random.normal(0, 1, self.n_ts)
        return ar_process


if __name__ == '__main__':
    # example usage
    ts_gen = TimeSeriesGenerator(n_periods=365, n_ts=3)
    sine_wave = ts_gen.generate_sine_wave(amplitude=10, period=365)
    random_walk = ts_gen.generate_random_walk(start_val=100)
    # ar_process = ts_gen.generate_autoregression(ar_params=[0.5, -0.3, 0.2])

    # plot the time series
    fig, axs = plt.subplots(3, figsize=(10, 10))
    axs[0].plot(sine_wave)
    axs[0].set_title('Sine Wave')
    axs[1].plot(random_walk)
    axs[1].set_title('Random Walk')
    # axs[2].plot(ar_process)
    # axs[2].set_title('Autoregressive Process')
    plt.tight_layout()
    plt.show()
