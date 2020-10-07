import torch
import numpy as np
import v2.mesh_op as mesh_op

from lie_learn.spaces import S2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def e2f(torch_edges, torch_faces):
    """ Calculate the intersection point from the edge to the face

    Args:
        torch_edges (Tensor): a m x 2 x 3 array, m is the number of cameras, 2 is two points, 3 is xyz coords
        torch_faces (Tensor): a n x 3 x 3 array, n is the number of faces, 3 is three points, 3 is xyz coords
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


class ProjectOnSphere:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.sgrid = self.make_sgrid(bandwidth, alpha=0, beta=0, gamma=0, grid_type='SOFT')

    def __call__(self, mesh):
        im = self.render_model(mesh, self.sgrid)
        return im

    def __repr__(self):
        return self.__class__.__name__ + '(bandwidth={0})'.format(self.bandwidth)

    @staticmethod
    def make_sgrid(b, alpha, beta, gamma, grid_type):
        theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
        sgrid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
        sgrid = sgrid.reshape((-1, 3))

        R = mesh_op.rotmat(alpha, beta, gamma, hom_coord=False)
        sgrid = np.einsum('ij,nj->ni', R, sgrid)

        return sgrid

    @staticmethod
    def render_model(mesh, sgrid):
        # Cast rays
        # triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
        index_tri, index_ray, loc = mesh.ray.intersects_id(
            ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
        loc = loc.reshape((-1, 3))  # fix bug if loc is empty

        # Each ray is in 1-to-1 correspondence with a grid point. Find the position of these points
        grid_hits = sgrid[index_ray]
        grid_hits_normalized = grid_hits / np.linalg.norm(grid_hits, axis=1, keepdims=True)

        # Compute the distance from the grid points to the intersection pionts
        dist = np.linalg.norm(grid_hits - loc, axis=-1)

        # For each intersection, look up the normal of the triangle that was hit
        normals = mesh.face_normals[index_tri]
        normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Construct spherical images
        dist_im = np.ones(sgrid.shape[0])
        dist_im[index_ray] = dist
        # dist_im = dist_im.reshape(theta.shape)

        # shaded_im = np.zeros(sgrid.shape[0])
        # shaded_im[index_ray] = normals.dot(light_dir)
        # shaded_im = shaded_im.reshape(theta.shape) + 0.4

        n_dot_ray_im = np.zeros(sgrid.shape[0])
        # n_dot_ray_im[index_ray] = np.abs(np.einsum("ij,ij->i", normals, grid_hits_normalized))
        n_dot_ray_im[index_ray] = np.einsum("ij,ij->i", normalized_normals, grid_hits_normalized)

        nx, ny, nz = normalized_normals[:, 0], normalized_normals[:, 1], normalized_normals[:, 2]
        gx, gy, gz = grid_hits_normalized[:, 0], grid_hits_normalized[:, 1], grid_hits_normalized[:, 2]
        wedge_norm = np.sqrt((nx * gy - ny * gx) ** 2 + (nx * gz - nz * gx) ** 2 + (ny * gz - nz * gy) ** 2)
        n_wedge_ray_im = np.zeros(sgrid.shape[0])
        n_wedge_ray_im[index_ray] = wedge_norm

        # Combine channels to construct final image
        # im = dist_im.reshape((1,) + dist_im.shape)
        im = np.stack((dist_im, n_dot_ray_im, n_wedge_ray_im), axis=0)

        return im
