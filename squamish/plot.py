import matplotlib

matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.cm as cm

# Get a color for each relevance type
color_palette_3 = cm.Set1([0, 1, 2], alpha=0.8)


def get_colors(N, classes=None):
    if classes is None:
        new_classes = np.zeros(N).astype(int)
        color = [color_palette_3[c.astype(int)] for c in new_classes]
    else:
        color = [color_palette_3[c.astype(int)] for c in classes]
    return color


def plot_bars(
    ax, importances, ticklabels=None, classes=None, numbering=True, tick_rotation=30
):
    """

    Parameters
    ----------
    ax:
        axis which the bars get drawn on
    importances:
        the vector of floating values determining the importance
    ticklabels: (optional)
        labels for each feature
    classes: (optional)
        relevance class for each feature, determines color
    numbering: bool
        Add feature index when using ticklabels
    tick_rotation:  int
        Amonut of rotation of ticklabels for easier readability.


    """
    N = len(importances)

    # Ticklabels
    if ticklabels is None:
        ticks = np.arange(N) + 1
    else:
        ticks = list(ticklabels)
        if numbering:
            for i in range(N):
                ticks[i] += " - {}".format(i + 1)

    # Interval sizes
    ind = np.arange(N) + 1
    width = 0.6
    height = importances

    # Bar colors
    color = get_colors(N, classes)

    # Plot the bars
    bars = ax.bar(
        ind,
        height,
        width,
        tick_label=ticks,
        align="center",
        edgecolor=["black"] * N,
        linewidth=1.3,
        color=color,
    )

    ax.set_xticklabels(ticks)
    if ticklabels is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=tick_rotation, ha="right")
    # ax.tick_params(rotation="auto")
    # Limit the y range to 0,1 or 0,L1
    ax.set_ylim([0, max(importances) * 1.1])

    ax.set_ylabel("relevance")
    ax.set_xlabel("feature")

    if classes is not None:
        relevance_classes = ["Irrelevant", "Weakly relevant", "Strongly relevant"]
        patches = []
        for i, rc in enumerate(relevance_classes):
            patch = mpatches.Patch(color=color_palette_3[i], label=rc)
            patches.append(patch)

        ax.legend(handles=patches)

    return bars


def plot_importances(importances, ticklabels=None, invert=False, classes=None):
    # Figure Parameters
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_bars(ax, importances, ticklabels=ticklabels, classes=classes)
    fig.autofmt_xdate()
    # Invert the xaxis for cases in which the comparison with other tools
    if invert:
        plt.gca().invert_xaxis()
    return fig


def plot_model_intervals(model, ticklabels=None):
    if model.feature_importances_ is not None:
        return plot_importances(
            model.feature_importances_,
            ticklabels=ticklabels,
            classes=model.relevance_classes_,
        )
    else:
        print("Intervals not computed. Try running fit() function first.")
