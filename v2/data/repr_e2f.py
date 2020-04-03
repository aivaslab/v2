import os
import sys
import glob
import pickle
import datetime
import math
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from v2.utils import conf

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def np_sin(x):
    """Since np.sin(np.pi) = 1e-16, this function will manually correct np.sin(np.pi) = 0

    Args:
        x:

    Returns:

    """
    sin_x = np.sin(x)
    sin_x[np.abs(sin_x) < 1e-8] = 0
    return sin_x


def all_factors(n):
    rst = []
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            rst.append(i)
    rst.append(n)
    return rst


# Current using sphere method, polar less, uniformly distributed
def linspace_sphere(u, v, r, polar=True):
    """ Generate points on a sphere based on the number of sampled points and the radius of the sphere

    Args:
        u (int): Sampling number of horizontal direction, the generated number is ranged from [0, 2pi]
        v (int): Sampling number of vertical direction. the generated number is ranged from [0, pi]
                 The arccos method used here is to reduce the polar effect
        r (float): The radius of the sphere
        polar (boolean): Include the polar point or not

    Returns:

    """
    theta = np.linspace(0, 1, u, endpoint=False)
    theta = 2 * np.pi * theta

    phi = np.linspace(0, 1, v)
    if not polar:
        phi = phi[~np.logical_or(phi == 0, phi == 1)]

    phi = np.arccos(1 - 2 * phi)

    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()

    # When phi is 0 or pi, the point is on the z-axis, and therefore x, y should be zero
    theta[np.logical_or(phi == 0, phi == np.pi)] = 0

    x = r * np_sin(phi) * np.cos(theta)
    y = r * np_sin(phi) * np_sin(theta)
    z = r * np.cos(phi)
    ray_origins = np.vstack((x, y, z))
    return ray_origins


def get_cams(s_col, s_row, p_col, p_row, d, polar=True):
    """A function to generate views such that:
        the center of each view is on the sphere, and the center will shoot a ray to the center of the object
        surrounding the view center, there will be multiple rays parallel to the center ray

    Args:
        s_col (int): Number of column views for the sphere
        s_row (int): Number of row views for the sphere
        p_col (int): Number of column pixels per view
        p_row (int): Number of row pixels per view
        d (float): The distance between two nearest view points
        polar (boolean): Include the polar point or not

    Returns:

    """
    s_view = linspace_sphere(u=s_col, v=s_row, r=0.5, polar=polar)
    s_d = -2 * s_view  # the direction vector shooting to the (0, 0, 0) with length 1, -s_view - s_view
    s_ray = np.dstack((s_view, -s_view)).transpose((1, 2, 0))

    # The orthonormal basis on the view surface
    x0 = s_view[0, :]
    y0 = s_view[1, :]
    z0 = s_view[2, :]

    vx = np.vstack((-y0, x0, np.zeros(x0.shape)))  # x basis, the one parallel to the z=0 surface
    vx[:, np.logical_and(x0 == 0, y0 == 0)] = np.vstack((np.ones(x0.shape),  # polar case
                                                         np.zeros(x0.shape),
                                                         np.zeros(x0.shape)))[:, np.logical_and(x0 == 0, y0 == 0)]

    vx = vx / np.linalg.norm(vx, 2, axis=0)

    vy = np.vstack((-z0, -z0 * y0 / x0, (x0 ** 2 + y0 ** 2) / x0))  # y basis
    vy[:, x0 == 0] = np.vstack((np.zeros((x0).shape), -z0, y0))[:, x0 == 0]  # x=0 case
    vy[:, np.logical_and(x0 == 0, y0 == 0)] = np.vstack((np.zeros(x0.shape),  # polar case
                                                         np.ones(x0.shape),
                                                         np.zeros(x0.shape)))[:, np.logical_and(x0 == 0, y0 == 0)]
    vy[:, x0 > 0] = -vy[:, x0 > 0]  # quadrant 1, 2 cases
    vy = vy / np.linalg.norm(vy, 2, axis=0)

    vz = s_view  # z basis, the one shooting to the origin
    vz = vz / np.linalg.norm(vz, 2, axis=0)

    # The meshgrid for the new orthonormal basis
    dx = np.linspace(-(p_col // 2) * d, p_col // 2 * d, p_col)
    dy = np.linspace(-(p_row // 2) * d, p_row // 2 * d, p_row)
    dx, dy = np.meshgrid(dx, dy)
    dx = dx.flatten()
    dy = dy.flatten()

    dvx = np.tile(vx, (len(dx), 1, 1)).transpose((1, 2, 0)) * dx
    dvy = np.tile(vy, (len(dy), 1, 1)).transpose((1, 2, 0)) * dy

    # Generate all view
    all_view = np.tile(s_view, (len(dx), 1, 1)).transpose((1, 2, 0)) + dvx + dvy
    all_ray = all_view + np.tile(s_d, (len(dx), 1, 1)).transpose((1, 2, 0))

    all_view = all_view.reshape((all_view.shape[0], all_view.shape[1] // s_col, s_col, p_row, p_col))
    all_view = all_view.transpose(0, 1, 3, 2, 4)
    all_ray = all_ray.reshape((all_ray.shape[0], all_ray.shape[1] // s_col, s_col, p_row, p_col))
    all_ray = all_ray.transpose(0, 1, 3, 2, 4)

    all_view = all_view.reshape((3, -1))
    all_ray = all_ray.reshape((3, -1))

    all_ray = np.dstack((all_view, all_ray)).transpose((1, 2, 0))

    return all_view, all_ray


def e2f(torch_edges, torch_faces):
    """ Calculate the intersection point from the edge to the face

    Args:
        torch_edges (Tensor): a m x 2 x 3 array, m is the number of cameras, 2 is two points, 3 is xyz coords
        torch_faces (Tensor): a n x 3 x 3 array, n is the number of faces, 3 is three points, 3 is xyz coords

    Returns:

    """
    # Get all intersected points using point-normal form
    # Reference, simple math for calculating the intersection of a plane and a line in a 3D space
    p0 = torch.mean(torch_faces, dim=1)
    e1 = torch_faces[:, 0, :] - torch_faces[:, 2, :]
    e2 = torch_faces[:, 1, :] - torch_faces[:, 2, :]
    n = torch.cross(e1, e2)
    l0 = torch_edges[:, 0, :]  # To be used in the next stage
    l = torch_edges[:, 1, :] - torch_edges[:, 0, :]

    p0 = p0.repeat(len(l0), 1, 1).permute(1, 0, 2)
    n = n.repeat(len(l0), 1, 1).permute(1, 0, 2)
    p0_l0_n = torch.sum((p0 - l0) * n, dim=2)

    # Calculate sin and cos
    l_repeat = l.repeat(len(n), 1, 1)
    # l_repeat_2norm = torch.norm(l_repeat, dim=2)  # all norm for my camera ray is 1
    n_2norm = torch.norm(n, dim=2)
    n_l_cross = torch.cross(n, l_repeat, dim=2)
    n_l_cross_2norm = torch.norm(n_l_cross, dim=2)
    n_l_dot = torch.sum(n * l_repeat, dim=2)
    n_l_sin = n_l_cross_2norm / n_2norm
    n_l_cos = n_l_dot / n_2norm

    # Keep calculating the intersected points
    l_n = torch.sum(l * n, dim=2)  # To be used in the next stage

    d = p0_l0_n / l_n
    d = torch.stack((d, d, d), dim=2)

    ip = d * l + l0  # Intersected points. To be used in the next stage
    bp = torch.ones(d.shape, dtype=d.dtype, device=d.device) * l + l0  # Boundary points.

    # Determine whether the intersected points is inside of the plane
    a = torch_faces[:, 0, :].repeat(len(l0), 1, 1).permute(1, 0, 2)
    b = torch_faces[:, 1, :].repeat(len(l0), 1, 1).permute(1, 0, 2)
    c = torch_faces[:, 2, :].repeat(len(l0), 1, 1).permute(1, 0, 2)

    v0 = c - a
    v1 = b - a
    v2 = ip - a

    v00 = torch.sum(v0 * v0, dim=2)
    v01 = torch.sum(v0 * v1, dim=2)
    v02 = torch.sum(v0 * v2, dim=2)
    v11 = torch.sum(v1 * v1, dim=2)
    v12 = torch.sum(v1 * v2, dim=2)

    denominator = v00 * v11 - v01 * v01
    u = (v11 * v02 - v01 * v12) / denominator
    v = (v00 * v12 - v01 * v02) / denominator

    inface = (u + v) <= (1 + 1e-6)

    inface[(u < (0 - 1e-6)) | (u > (1 + 1e-6))] = False
    inface[(v < (0 - 1e-6)) | (v > (1 + 1e-6))] = False

    ip2l0_d = torch.norm(ip - l0, dim=2)
    ip2l0_d[~inface] = 1  # equals to the diameter of the sphere
    ip2l0_d[l_n == 0] = 1  # equals to the diameter of the sphere

    # Get minimum distance
    ip2l0_d_min, ip2l0_d_argmin = torch.min(ip2l0_d, dim=0)

    # Get the coords of the intersected points with minimum distance
    ip2l0 = ip[ip2l0_d_argmin]
    bp2l0 = bp[ip2l0_d_argmin]
    ip2l0[~inface[ip2l0_d_argmin]] = bp2l0[~inface[ip2l0_d_argmin]]
    ip2l0[(l_n == 0)[ip2l0_d_argmin]] = bp2l0[(l_n == 0)[ip2l0_d_argmin]]

    n_l_sin = n_l_sin[ip2l0_d_argmin]
    n_l_cos = n_l_cos[ip2l0_d_argmin]

    ip2l0_diag = torch.diagonal(ip2l0, dim1=0, dim2=1).transpose(1, 0)
    ip2l0_sin = torch.diagonal(n_l_sin)
    ip2l0_cos = torch.diagonal(n_l_cos)

    return ip2l0_diag, ip2l0_d_min, ip2l0_sin, ip2l0_cos


def e2f_stepped(ray_edges, mesh_triangles, face_interval, edge_interval):
    """ GPU has very limited memory, so I have to split rays and triangles into small batches

    Args:
        ray_edges:
        mesh_triangles:
        face_interval (int): The interval per mini-batch to split triangles
        edge_interval (int): The interval per mini-batch to split rays

    Returns:

    """
    face_steps = int(np.ceil(mesh_triangles.size()[0] / face_interval))
    edge_steps = int(np.ceil(ray_edges.size()[0] / edge_interval))
    ip2l0_diag_all = torch.zeros(face_steps, ray_edges.size()[0], 3, device=DEVICE)
    ip2l0_d_min_all = torch.zeros(face_steps, ray_edges.size()[0], device=DEVICE)
    ip2l0_sin_all = torch.zeros(face_steps, ray_edges.size()[0], device=DEVICE)
    ip2l0_cos_all = torch.zeros(face_steps, ray_edges.size()[0], device=DEVICE)

    print('Total steps: %d' % face_steps)
    for i in range(face_steps):
        for j in range(edge_steps):
            ip2l0_diag, ip2l0_d_min, ip2l0_sin, ip2l0_cos = e2f(
                ray_edges[j * edge_interval: min((j + 1) * edge_interval, ray_edges.size()[0])],
                mesh_triangles[i * face_interval:(i + 1) * face_interval]
            )
            ip2l0_diag_all[i,
            j * edge_interval:min((j + 1) * edge_interval, ray_edges.size()[0])] = ip2l0_diag
            ip2l0_d_min_all[i,
            j * edge_interval:min((j + 1) * edge_interval, ray_edges.size()[0])] = ip2l0_d_min
            ip2l0_sin_all[i,
            j * edge_interval:min((j + 1) * edge_interval, ray_edges.size()[0])] = ip2l0_sin
            ip2l0_cos_all[i,
            j * edge_interval:min((j + 1) * edge_interval, ray_edges.size()[0])] = ip2l0_cos

    ip2l0_d_min, ip2l0_d_argmin_all = torch.min(ip2l0_d_min_all, dim=0)
    ip2l0_sin, _ = torch.min(ip2l0_sin_all, dim=0)
    ip2l0_cos, _ = torch.min(ip2l0_cos_all, dim=0)

    ip2l0_d_argmin_all = ip2l0_d_argmin_all.repeat(3, 1).transpose(0, 1).unsqueeze(0)
    ip2l0_diag = ip2l0_diag_all.gather(0, ip2l0_d_argmin_all).squeeze()

    return ip2l0_diag, ip2l0_d_min, ip2l0_sin, ip2l0_cos


def get_modelnet_pickle_files(data_dir):
    def sort_func(x):
        no = os.path.splitext(os.path.basename(x))[0].split('_')[-1]
        ca = '_'.join(os.path.splitext(os.path.basename(x))[0].split('_')[:-1])
        no = int(no)
        return ca, no

    train_files = glob.glob(os.path.join(data_dir, '*', 'train', '*.pickle'))
    test_files = glob.glob(os.path.join(data_dir, '*', 'test', '*.pickle'))

    return sorted(train_files + test_files, key=sort_func)


def render_dsc_img(ip2l0_d_min, ip2l0_sin, ip2l0_cos, row_num, col_num, cam_settings, render_dir):
    """ d is the distance channel, s is the sine channel, c is the cosine channel --> dsc
    """
    d_img = np.reshape(ip2l0_d_min.cpu().numpy(), (row_num, col_num))
    s_img = np.reshape(ip2l0_sin.cpu().numpy(), (row_num, col_num))
    c_img = np.reshape(ip2l0_cos.cpu().numpy(), (row_num, col_num))
    img = np.dstack([d_img, s_img, c_img])
    if render_dir is not None:
        if not os.path.exists(os.path.dirname(render_dir)):
            os.makedirs(os.path.dirname(render_dir))
        with open('%s' % render_dir, 'wb+') as f:
            pickle.dump(img, f)
    else:
        plt.clf()
        plt.imshow(img, vmin=0, vmax=1, cmap='gray')
        with open('/home/tengyu/Desktop/%s_img.png' % cam_settings, 'wb+') as f:
            plt.savefig(f, bbox_inches='tight')


def generate_v2():
    nv_start = 1  # number of views in the start condition
    side = 200  # number of columns and rows per view in the start condition,  224 for multi-view, 128 for sphere
    c = nv_start * side ** 2  # total number of pixels, 4 is 4 views, 50 * 50 is the pixels per view
    factors = all_factors(side)  # side is the number of col/row per view

    for i, fc in enumerate(factors):
        s_col = nv_start * side // fc
        s_row = side // fc + 2
        p_col = fc
        p_row = fc
        d = 1 / side

        s_col = 32
        s_row = 32 + 2
        p_col = 1
        p_row = 1
        d = 1 / side

        cam_settings = '%d_%d_%d_%d_%.4f' % (s_col, s_row - 2, p_col, p_row, d)
        print('Cam settings: %s' % cam_settings)
        # Cam Settings
        np_cams_xyz, np_cams_edges = get_cams(s_col, s_row, p_col, p_row, d, polar=False)

        # Data Settings
        data_dir = conf.ModelNet40FacesNP_DIR
        all_pk_files = get_modelnet_pickle_files(data_dir)

        torch_cams_edges = torch.from_numpy(np_cams_edges).to(DEVICE)

        start = datetime.datetime.now()
        time_i = 0
        for j, np_faces_xyz_pk in enumerate(all_pk_files):
            render_dir = np_faces_xyz_pk.replace('ModelNet40FacesNP', 'ModelNet40Angle_c%d' % c)
            render_dir = render_dir.replace('.pickle', '/%s.pickle' % cam_settings)

            if os.path.isfile(render_dir):
                print('Cam %d/%d, Mesh %d/%d --- Already rendered: %s' % (
                    i + 1, len(factors), j + 1, len(all_pk_files), render_dir))
                continue
            else:
                print('Cam %d/%d, Mesh %d/%d --- Rendering: %s' % (
                    i + 1, len(factors), j + 1, len(all_pk_files), render_dir))
                time_i += 1

            end = datetime.datetime.now()
            print('%s / %s ' % (str(end - start), str((end - start) / time_i * (len(all_pk_files) - j))))

            with open(np_faces_xyz_pk, 'rb') as f:
                np_faces_xyz = pickle.load(f)
            torch_faces_xyz = torch.from_numpy(np_faces_xyz).to(DEVICE)
            ip2l0_diag, ip2l0_d_min, ip2l0_sin, ip2l0_cos = e2f_stepped(torch_cams_edges, torch_faces_xyz,
                                                                        face_interval=1024,
                                                                        edge_interval=256)
            # Rendering depth, sine, cosine
            render_dsc_img(ip2l0_d_min, ip2l0_sin, ip2l0_cos, side, nv_start * side, cam_settings, render_dir)
        break


def main():
    generate_v2()


if __name__ == '__main__':
    main()
