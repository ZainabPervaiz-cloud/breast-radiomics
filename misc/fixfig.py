"""
Python port of fixfig.m
Set figure properties for publication-ready output.
"""
import matplotlib.pyplot as plt
import matplotlib


def fixfig(w_cm=18, h_cm=16, fontname='monospace', fontsize=8):
    """
    Apply standard figure formatting (equivalent to MATLAB fixfig.m).

    Parameters
    ----------
    w_cm : float
        Figure width in centimeters.
    h_cm : float
        Figure height in centimeters.
    fontname : str
        Font name for tick labels.
    fontsize : int
        Font size for tick labels.
    """
    fig = plt.gcf()
    # Convert cm to inches
    w_in = w_cm / 2.54
    h_in = h_cm / 2.54
    fig.set_size_inches(w_in, h_in)

    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
        item.set_fontfamily(fontname)

    plt.tight_layout()
