from mpl_toolkits.mplot3d import Axes3D, art3d  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib
import seaborn as sns
import torch
import pickle
import scipy as sp
import numpy as np
import pandas as pd
import imageio
import glob
import os

from v2.data import v2gen
from v2.utils import conf

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def plt_v2_config(v2, convh, mesh, ax):
    # ax = Axes3D(plt.figure(figsize=(10, 10)))
    cube_d = v2.r
    ax.set_xlim(-cube_d, cube_d)
    ax.set_ylim(-cube_d, cube_d)
    ax.set_zlim(-cube_d, cube_d)
    ax.scatter(v2.ray_origins[:, 0], v2.ray_origins[:, 1], v2.ray_origins[:, 2])

    if convh:
        convh_v2_p = v2.convh_v2_p[~(v2.convh_v2_d == 2 * v2.r).flatten()]
        ax.scatter(convh_v2_p[:, 0], convh_v2_p[:, 1], convh_v2_p[:, 2])
        tris = v2.convex_hull.triangles

    else:
        mesh_v2_p = v2.mesh_v2_p[~(v2.mesh_v2_d == 2 * v2.r).flatten()]
        ax.scatter(mesh_v2_p[:, 0], mesh_v2_p[:, 1], mesh_v2_p[:, 2])
        tris = v2.mesh.triangles

    if mesh:
        for vtx in tris:
            tri = art3d.Poly3DCollection([vtx])
            tri.set_color(colors.rgb2hex(sp.rand(3)))
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)

    plt.axis('off')


def plt_v2_repr(v2, channel, ax):
    reprs = [v2.mesh_v2_d, v2.mesh_v2_s, v2.mesh_v2_c,
             v2.convh_v2_d, v2.convh_v2_s, v2.convh_v2_c]


    if channel == 0 or channel == 3:  # Depth channel
        ax.imshow(reprs[channel], vmin=0, vmax=2 * v2.r, cmap='gray')
    else:  # Sine, Cosine channel
        ax.imshow(reprs[channel], vmin=0, vmax=1, cmap='gray')

    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()


def fig4():
    cams = sorted(glob.glob('/home/tengyu/Desktop/cvpr_workshop/fig4/cam/*'), key=lambda x: int(os.path.basename(x).split('_')[0]))
    ds = sorted(glob.glob('/home/tengyu/Desktop/cvpr_workshop/fig4/d/*'), key=lambda x: int(os.path.basename(x).split('_')[0]))

    def draw_line(i, cam, d):
        ax1 = fig.add_subplot(spec2[i, 0])
        ax1.imshow(imageio.imread(cam))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, chr(97 + i), transform=ax1.transAxes, fontsize=15,
                 verticalalignment='top')

        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

        ax1.axis('off')

        ax2 = fig.add_subplot(spec2[i, 1:])
        ax2.imshow(imageio.imread(d))
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_aspect('equal')

        ax2.axis('off')

    fig = plt.figure(figsize=(9.6, 12.8))
    spec2 = gridspec.GridSpec(ncols=4, nrows=6, figure=fig)
    spec2.update(wspace=0, hspace=0)
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(cams)):
        cam = cams[i]
        d = ds[i]

        draw_line(i, cam, d)

    plt.savefig('fig4.png', bbox_inches='tight')


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(
        rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9)
    )
    bleed = 0
    fig.subplots_adjust(
        left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed)
    )

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
    plt.show()


def plt_confusion_matrix():
    df = pd.read_csv(os.path.join(conf.RST_DIR, 'tmp.csv'), index_col=0)
    plt.figure(figsize=(12, 10), dpi=100)
    sns.heatmap(df, cmap="YlGnBu")
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('cm_heatmap.png')
    # plt.show()


def depth_analysis():
    fig, axes = plt.subplots(2, 3)
    mt = np.load(conf.PROJ_DATA_DIR + '/mt.npy')
    s2cnn = np.load(conf.PROJ_DATA_DIR + '/s2cnn.npy')
    trimesh = np.load(conf.PROJ_DATA_DIR + '/trimesh.npy')
    convmt = np.load(conf.PROJ_DATA_DIR + '/convmt.npy')
    convs2cnn = np.load(conf.PROJ_DATA_DIR + '/convs2cnn.npy')
    convtrimesh = np.load(conf.PROJ_DATA_DIR + '/convtrimesh.npy')

    allmesh = [mt, s2cnn, trimesh, convmt, convs2cnn, convtrimesh]
    for i, m in enumerate(allmesh):
        axes = axes.flatten()
        ax = axes[i]
        ax.imshow(m, cmap='gray', vmin=0, vmax=2)
        ax.set_axis_off()

    plt.show()


def main():
    depth_analysis()


if __name__ == '__main__':
    main()
