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


def plot_cams_faces(ip2l0_diag, np_edges, cam_settings, np_faces=None):
    l0 = np_edges[:, 0, :]

    # matplotlib.use('TKAGG')
    ax3 = Axes3D(plt.figure(figsize=(10, 10)))

    # Draw cameras
    cube_d = 0.5
    ax3.set_xlim(-cube_d, cube_d)
    ax3.set_ylim(-cube_d, cube_d)
    ax3.set_zlim(-cube_d, cube_d)
    ax3.scatter(l0[:, 0], l0[:, 1], l0[:, 2])

    # Draw intersected points
    a = (ip2l0_diag[:, 0] >= 0.3) | (ip2l0_diag[:, 0] <= -0.3)
    b = (ip2l0_diag[:, 1] >= 0.3) | (ip2l0_diag[:, 1] <= -0.3)
    c = (ip2l0_diag[:, 2] >= 0.3) | (ip2l0_diag[:, 2] <= -0.3)
    d = a | b | c
    ip2l0_diag = ip2l0_diag[~d, :]

    ax3.scatter(ip2l0_diag[:, 0], ip2l0_diag[:, 1], ip2l0_diag[:, 2])
    # major_ticks = np.arange(-0.5, 0.5, 1/31)
    # ax3.set_xticks(major_ticks)
    # ax3.set_yticks(major_ticks)
    # ax3.set_zticks(major_ticks)
    ax3.grid(True)
    # plt.set_linewidth(1/31)

    # Draw triangle faces
    if np_faces is not None:
        for vtx in np_faces:
            # vtx = np_faces
            tri = art3d.Poly3DCollection([vtx])
            tri.set_color(colors.rgb2hex(sp.rand(3)))
            tri.set_edgecolor('k')
            ax3.add_collection3d(tri)

    # plt.show()
    parameters = cam_settings.split('_')
    m, n, x, y = list(map(int, parameters[:4]))
    d = float(parameters[-1])
    # plt.title('m=%d\nn=%d\nx=%d\ny=%d\nd=%.2f' % (m, n, x, y, d), fontsize=20, y=0.5, x=0.1)
    plt.axis('off')
    # plt.show()

    with open('/home/tengyu/Desktop/cvpr_workshop/fig2/cam/%s_cam.png' % cam_settings, 'wb+') as f:
        plt.savefig(f, bbox_inches='tight')


def plot_d_img(ip2l0_d_min, row_num, col_num, cam_settings):
    d_img = np.reshape(ip2l0_d_min.cpu().numpy(), (row_num, col_num))
    plt.clf()
    plt.imshow(d_img, vmin=0, vmax=1, cmap='gray')
    # with open('/home/tengyu/Desktop/%s_img.png' % cam_settings, 'wb+') as f:
    #     plt.savefig(f, bbox_inches='tight')
    ax = plt.gca()
    # ax.set_xticks(np.arange(0, 25 * 9, 25))
    # ax.set_yticks(np.arange(0, 10 * 6, 10))
    plt.axis('off')
    plt.show()
    # with open('/home/tengyu/Desktop/cvpr_workshop/fig4/d/%s_cam.png' % cam_settings, 'wb+') as f:
    #     plt.imsave(f, d_img, vmin=0, vmax=1, cmap='gray')


def draw_designated_rep():
    # ToDo: should allow non-integer case, otherwise the evolving process is not smooth enough
    # side = 840  # number of columns and rows per view in the start condition,
    # factors = all_factors(side)  # side is the number of col/row per view
    #
    # for fc in factors:
    #     s_col = side // fc
    #     s_row = side // fc + 2
    #     p_col = fc
    #     p_row = fc
    #     d = 1 / 200 / 2

    cvpr_evolving_obj = 401
    cvpr_evolving_cam = [
        [4, 1 + 2, 50, 50, 0.02],
        [8, 2 + 2, 25, 25, 0.02],
        [20, 5 + 2, 10, 10, 0.02],
        [40, 10 + 2, 5, 5, 0.02],
        [100, 25 + 2, 2, 2, 0.02],
        [200, 50 + 2, 1, 1, 0.02]
    ]

    cvpr_v2_obj = 0
    cvpr_v2_cam = [
        [4, 1 + 2, 50, 50, 0.02],
        [8, 2 + 2, 25, 25, 0.02],
        [20, 5 + 2, 10, 10, 0.02],
        [40, 10 + 2, 5, 5, 0.02],
        [100, 25 + 2, 2, 2, 0.02],
        [200, 50 + 2, 1, 1, 0.02]
    ]

    # s_col = 200
    # s_row = 52
    # p_col = 1
    # p_row = 1
    # d = 0.02

    fig_2_cam = [
        [4, 1 + 2, 25, 25, 1 / 50],
        [4, 1 + 2, 25, 25, 1 / 100],
        [4, 4 + 2, 25, 25, 1 / 100],
        [8, 4 + 2, 25, 25, 1 / 100],
        [8, 4 + 2, 25, 25, 1 / 200],
        [8, 8 + 2, 10, 10, 1 / 200],
        [16, 16 + 2, 5, 5, 1 / 200],
        [16, 16 + 2, 1, 1, 1 / 32],
        [16, 16 + 2, 1, 1, 1 / 200],

    ]


    for cam in cvpr_evolving_cam[:]:
        s_col = 16
        s_row = 16 + 2
        p_col = 5
        p_row = 5
        d = 1 / 16

        # s_col, s_row, p_col, p_row, d = cam
        cam_settings = '%d_%d_%d_%d_%.04f' % (s_col, s_row - 2, p_col, p_row, d)
        # print('Cam settings: %s' % cam_settings)
        # Cam Settings
        np_cams_xyz, np_cams_edges = v2_generator.get_cams(s_col, s_row, p_col, p_row, d, polar=False)

        # Data Settings
        data_dir = conf.ModelNet40FacesNP_DIR
        all_pk_files = v2_generator.get_modelnet_pickle_files(data_dir)

        torch_cams_edges = torch.from_numpy(np_cams_edges).to(DEVICE)

        j = 80

        # for i, e in enumerate(all_pk_files):
        #     if 'chair' not in e:
        #         continue
        #
        #     if 'test' not in e:
        #         continue

        # print(i)
        np_faces_xyz_pk = all_pk_files[j]

        with open(np_faces_xyz_pk, 'rb') as f:
            np_faces_xyz = pickle.load(f)

        torch_faces_xyz = torch.from_numpy(np_faces_xyz).to(DEVICE)
        ip2l0_diag, ip2l0_d_min, ip2l0_sin, ip2l0_cos = v2_generator.e2f_stepped(torch_cams_edges, torch_faces_xyz, face_interval=1024, edge_interval=1024)
        plot_cams_faces(np.reshape(ip2l0_diag.cpu().numpy(), (-1, 3)),
                        np_cams_edges, cam_settings, np_faces_xyz)

        # Rendering depth pickle
        # plot_d_img(ip2l0_d_min, (s_row - 2) * p_row, s_col * p_col, cam_settings)
        break


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
