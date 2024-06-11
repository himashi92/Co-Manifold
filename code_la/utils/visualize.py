import os
import argparse
import sys

import geoopt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def add_geodesic_grid(ax: plt.Axes, manifold: geoopt.Stereographic, line_width=0.1):

    # define geodesic grid parameters
    N_EVALS_PER_GEODESIC = 10000
    STYLE = "--"
    COLOR = "gray"
    LINE_WIDTH = line_width

    # get manifold properties
    K = manifold.k.item()
    R = manifold.radius.item()

    # get maximal numerical distance to origin on manifold
    if K < 0:
        # create point on R
        r = torch.tensor((R, 0.0), dtype=manifold.dtype)
        # project point on R into valid range (epsilon border)
        r = manifold.projx(r)
        # determine distance from origin
        max_dist_0 = manifold.dist0(r).item()
    else:
        max_dist_0 = np.pi * R
    # adjust line interval for spherical geometry
    circumference = 2*np.pi*R

    # determine reasonable number of geodesics
    # choose the grid interval size always as if we'd be in spherical
    # geometry, such that the grid interpolates smoothly and evenly
    # divides the sphere circumference
    n_geodesics_per_circumference = 4 * 6  # multiple of 4!
    n_geodesics_per_quadrant = n_geodesics_per_circumference // 2
    grid_interval_size = circumference / n_geodesics_per_circumference
    if K < 0:
        n_geodesics_per_quadrant = int(max_dist_0 / grid_interval_size)

    # create time evaluation array for geodesics
    if K < 0:
        min_t = -1.2*max_dist_0
    else:
        min_t = -circumference/2.0
    t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC)[:, None]
    # define a function to plot the geodesics
    def plot_geodesic(gv):
        ax.plot(*gv.t().numpy(), STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # define geodesic directions
    u_x = torch.tensor((0.0, 1.0))
    u_y = torch.tensor((1.0, 0.0))

    # add origin x/y-crosshair
    o = torch.tensor((0.0, 0.0))
    if K < 0:
        x_geodesic = manifold.geodesic_unit(t, o, u_x)
        y_geodesic = manifold.geodesic_unit(t, o, u_y)
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
    else:
        # add the crosshair manually for the sproj of sphere
        # because the lines tend to get thicker if plotted
        # as done for K<0
        ax.axvline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)
        ax.axhline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # add geodesics per quadrant
    for i in range(1, n_geodesics_per_quadrant):
        i = torch.as_tensor(float(i))
        # determine start of geodesic on x/y-crosshair
        x = manifold.geodesic_unit(i*grid_interval_size, o, u_y)
        y = manifold.geodesic_unit(i*grid_interval_size, o, u_x)

        # compute point on geodesics
        x_geodesic = manifold.geodesic_unit(t, x, u_x)
        y_geodesic = manifold.geodesic_unit(t, y, u_y)

        # x_geodesic = x_geodesic.detach().numpy()
        # y_geodesic = y_geodesic.detach().numpy()
        # plot geodesics
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
        if K < 0:
            plot_geodesic(-x_geodesic)
            plot_geodesic(-y_geodesic)


def add_euclidean_grid(ax, grid_spacing=1.0, color='gray', linestyle='--', line_width=1.0):
    """
    Add a grid to a Matplotlib axis for visualizing points in Euclidean space.

    Args:
        ax (matplotlib.axes._axes.Axes): The Matplotlib axis to which the grid will be added.
        grid_spacing (float, optional): The spacing between grid lines. Default is 1.0.
        color (str, optional): The color of the grid lines. Default is 'gray'.
        linestyle (str, optional): The linestyle of the grid lines. Default is '--'.
        line_width (float, optional): The line width of the grid lines. Default is 1.0.
    """
    # Determine the range of x and y values based on the axis limits
    xlim = -2.1, 2.1
    ylim = -2.1, 2.1

    x_min, x_max = xlim
    y_min, y_max = ylim

    # Create grid lines for the x-axis
    x_grid = np.arange(np.floor(x_min), np.ceil(x_max), grid_spacing)
    for x in x_grid:
        ax.axvline(x, color=color, linestyle=linestyle, linewidth=line_width, alpha=0.5)

    # Create grid lines for the y-axis
    y_grid = np.arange(np.floor(y_min), np.ceil(y_max), grid_spacing)
    for y in y_grid:
        ax.axhline(y, color=color, linestyle=linestyle, linewidth=line_width, alpha=0.5)

    # Set axis limits to match the original data
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def save_plots(outputs1, zs_H, hyp, outputs2, zs_E, save_path, i):
    embeddings1 = zs_H[: ,0].detach().cpu()
    embeddings_array1 = embeddings1.reshape(-1, 2).numpy()
    segs1 = outputs1[0].cpu().detach().numpy() > 0.5

    embeddings2 = zs_E[:, 0].detach().cpu()
    embeddings_array2 = embeddings2.reshape(-1, 2).numpy()
    segs2 = outputs2[0].cpu().detach().numpy() > 0.5

    et1 = segs1[0]
    labelmap1 = np.zeros(segs1[0].shape)
    labelmap1[et1] = 1
    pred1 = labelmap1

    et2 = segs2[0]

    labelmap2 = np.zeros(segs2[0].shape)
    labelmap2[et2] = 1

    pred2 = labelmap2

    voxelarray1 = pred1
    voxelarray2 = pred2

    colors1 = np.empty(voxelarray1.shape, dtype=object)
    colors1[pred1 == 1] = 'red'

    colors2 = np.empty(voxelarray2.shape, dtype=object)
    colors2[pred2 == 1] = 'red'

    # Create subplots with 3D image plots
    fig = plt.figure(figsize=(18, 6))

    # 3D Image Plot for pred 1
    ax1 = fig.add_subplot(141, projection='3d')  # 3D subplot for pred
    ax1.voxels(voxelarray1, facecolors=colors1, edgecolor='k', linewidth=0 ,)
    ax1.set_title("Prediction from Hyperbolic Probabilistic VNet")

    # Plot for h-embeddings
    ax2 = fig.add_subplot(142)
    circle = plt.Circle((0, 0), 1, fill=False, color="b")
    add_geodesic_grid(ax2, hyp, 0.5)
    ax2.add_artist(circle)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect("equal")
    ax2.scatter(embeddings_array1[:, 0], embeddings_array1[:, 1], alpha=0.3)
    ax2.set_title("Hyperbolic Embeddings")

    # 3D Image Plot for pred 2
    ax3 = fig.add_subplot(143, projection='3d')  # 3D subplot for pred
    ax3.voxels(voxelarray2, facecolors=colors2, edgecolor='k', linewidth=0, )
    ax3.set_title("Prediction from Euclidean Probabilistic VNet")

    # Plot for e-embeddings
    ax4 = fig.add_subplot(144)
    ax4.scatter(embeddings_array2[:, 0], embeddings_array2[:, 1], alpha=0.3)
    add_euclidean_grid(ax4, grid_spacing=0.2, linestyle='--', color='gray', line_width=0.1)
    ax4.set_title("Euclidean Embeddings")

    # Add spacing between subplots
    plt.tight_layout()
    # Show the plots
    plt.savefig(save_path + f"/{i}_plot.png")

    plt.close(fig)


def save_plots_all(gt, outputs1, zs_H, hyp, outputs2, zs_E, save_path, i):
    embeddings1 = zs_H[: ,0].detach().cpu()
    embeddings_array1 = embeddings1.reshape(-1, 2).numpy()
    segs1 = outputs1[0].cpu().detach().numpy() > 0.5

    embeddings2 = zs_E[:, 0].detach().cpu()
    embeddings_array2 = embeddings2.reshape(-1, 2).numpy()
    segs2 = outputs2[0].cpu().detach().numpy() > 0.5

    gt_segs = gt[0].detach().cpu().numpy() > 0.5
    #gt_segs = gt_segs.unsqueeze(0)

    #print(f"OUT : {segs1.shape}, GT : {gt_segs.shape}")
    et_gt = gt_segs
    labelmap_gt = np.zeros(gt_segs.shape)
    labelmap_gt[et_gt] = 1
    label = labelmap_gt

    et1 = segs1[0]
    labelmap1 = np.zeros(segs1[0].shape)
    labelmap1[et1] = 1
    pred1 = labelmap1

    et2 = segs2[0]
    labelmap2 = np.zeros(segs2[0].shape)
    labelmap2[et2] = 1

    pred2 = labelmap2

    voxelarray1 = pred1
    voxelarray2 = pred2
    voxelarray3 = label

    colors1 = np.empty(voxelarray1.shape, dtype=object)
    colors1[pred1 == 1] = 'red'

    colors2 = np.empty(voxelarray2.shape, dtype=object)
    colors2[pred2 == 1] = 'red'

    colors3 = np.empty(voxelarray3.shape, dtype=object)
    colors3[label == 1] = 'red'

    # Create subplots with 3D image plots
    fig = plt.figure(figsize=(18, 6))
    # 3D Image Plot for pred 1
    ax1 = fig.add_subplot(151, projection='3d')  # 3D subplot for pred
    ax1.voxels(voxelarray3, facecolors=colors3, edgecolor='k', linewidth=0,)
    ax1.set_title("Ground Truth")

    # 3D Image Plot for pred 1
    ax1 = fig.add_subplot(152, projection='3d')  # 3D subplot for pred
    ax1.voxels(voxelarray1, facecolors=colors1, edgecolor='k', linewidth=0 ,)
    ax1.set_title("Prediction from Hyperbolic Probabilistic VNet")

    # Plot for h-embeddings
    ax2 = fig.add_subplot(153)
    circle = plt.Circle((0, 0), 1, fill=False, color="b")
    add_geodesic_grid(ax2, hyp, 0.5)
    ax2.add_artist(circle)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect("equal")
    ax2.scatter(embeddings_array1[:, 0], embeddings_array1[:, 1], alpha=0.3)
    ax2.set_title("Hyperbolic Embeddings")

    # 3D Image Plot for pred 2
    ax3 = fig.add_subplot(154, projection='3d')  # 3D subplot for pred
    ax3.voxels(voxelarray2, facecolors=colors2, edgecolor='k', linewidth=0, )
    ax3.set_title("Prediction from Euclidean Probabilistic VNet")

    # Plot for e-embeddings
    ax4 = fig.add_subplot(155)
    ax4.scatter(embeddings_array2[:, 0], embeddings_array2[:, 1], alpha=0.3)
    add_euclidean_grid(ax4, grid_spacing=0.2, linestyle='--', color='gray', line_width=0.1)
    ax4.set_title("Euclidean Embeddings")

    # Add spacing between subplots
    plt.tight_layout()
    # Show the plots
    plt.savefig(save_path + f"/{i}_plot.png")

    plt.close(fig)

