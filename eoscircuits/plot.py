"""Plotting Modules for EOSCircuits

This module uses :code:`olftrans.plot` as basis and adds additional
features to handle data formats that are returned by :code:`NeuroDriver`
"""
import typing as tp
import numpy as np
import matplotlib.pyplot as plt
from olftrans.plot import plot_mat


class EOSPlotterException(Exception):
    """Base EOS plotter exception"""

    pass


def plot_data(data: dict, markersize: int = None, color="k", **kwargs) -> plt.Axes:
    """Plot Continuous Valued Data

    Plots data returned by NeuroDriver's :code:`OutputRecorder`
    where each the data is a dictionary of dictionaries, where the first
    dictionary is keyed by node_id and the second dictionary is keyed
    by 'data' and 'time'
    """
    matrix = []
    for name, d in data.items():
        if not "data" in d and "time" in d:
            raise EOSPlotterException(
                f"Data for node {name} is not compatible with required format, "
                "data mush have both 'data' and 'time' fields. Did you mean "
                "to call plot_spikes?"
            )
        matrix.append(d["data"])
    matrix = np.vstack(matrix)
    return plot_mat(mat=matrix, **kwargs)


def plot_spikes(
    spikes: dict,
    ax: plt.Axes = None,
    markersize: int = None,
    color: tp.Union[str, tp.Any] = "k",
) -> plt.Axes:
    """Plot Spikes returned by NeuroDriver's OutputRecorder"""
    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot()

    for n, (name, ss) in enumerate(spikes.items()):
        if "data" not in ss:
            raise EOSPlotterException(f"'data' field missing for node {name}")
        if "data" in ss and "time" in ss:
            raise EOSPlotterException(
                f"Data for node {name} is not compatible with required format, "
                "data mush only have 'data' field. Did you mean "
                "to call plot_data?"
            )
        if len(ss["data"]) > 0:
            ax.plot(
                ss["data"],
                np.full(len(ss["data"]), n),
                "|",
                c=color,
                markersize=markersize,
            )
    return ax
