
import matplotlib
from .plot_profiles import *

PARAMS = {
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 8,
    'legend.fontsize': 6,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
}

matplotlib.rcParams.update(PARAMS)

__all__ = plot_profiles.__all__
