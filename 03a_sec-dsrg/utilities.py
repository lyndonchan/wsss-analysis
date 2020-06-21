import numpy as np
import matplotlib.pyplot as plt

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, rot_angle=0):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='afmhot', vmin=0.0, vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    # ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    if rot_angle == 45:
        ax.set_xticks(np.arange(AUC.shape[1]), minor=False)
        ax.set_xticklabels(xticklabels, minor=False, rotation=rot_angle, horizontalalignment='left')
    else:
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(xticklabels, minor=False, rotation=rot_angle)
    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    # plt.title(title, y=1.08, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    axis_offset = -0.012*AUC.shape[0] + 1.436
    ax.xaxis.set_label_coords(.5, axis_offset)

    # Remove last blank column
    # plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    ax.axis('equal')
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    # plt.colorbar(c)

    # Add text in each cell
    cell_font = 10 # math.ceil(AUC.shape[1] * 10 / 28)
    show_values(c, fontsize=cell_font)

    # Proper orientation (origin at the top left instead of bottom left)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    ax.axis('tight')
    fig_len = 40 / 28 * AUC.shape[0]

    fig.set_size_inches(cm2inch(fig_len, fig_len))