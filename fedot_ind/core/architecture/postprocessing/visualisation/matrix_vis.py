from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2

cols = plt.get_cmap('tab10').colors

def plot_2d(m, title=""):
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

def plot_wcorr(minimum=None, maximum=None):
    """Plots the w-correlation matrix for the decomposed time series.

    """
    if minimum is None:
        minimum = 0
    if maximum is None:
        maximum = d

    if Wcorr is None:
        calc_wcorr()

    ax = plt.imshow(Wcorr)
    plt.xlabel(r"$\tilde{F}_i$")
    plt.ylabel(r"$\tilde{F}_j$")
    plt.colorbar(ax.colorbar, fraction=0.045)
    ax.colorbar.set_label("$W_{i,j}$")
    plt.clim(0, 1)

    # For plotting purposes:
    if maximum == d:
        max_range = d - 1
    else:
        max_range = maximum

    plt.xlim(minimum - 0.5, max_range + 0.5)
    plt.ylim(max_range + 0.5, minimum - 0.5)