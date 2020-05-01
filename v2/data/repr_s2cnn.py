import numpy as np

from v2.utils import mesh_op


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


def make_sgrid(b, alpha, beta, gamma, grid_type):
    from lie_learn.spaces import S2

    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    sgrid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
    sgrid = sgrid.reshape((-1, 3))

    R = mesh_op.rotmat(alpha, beta, gamma, hom_coord=False)
    sgrid = np.einsum('ij,nj->ni', R, sgrid)

    return sgrid


class ProjectOnSphere:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.sgrid = make_sgrid(bandwidth, alpha=0, beta=0, gamma=0, grid_type='SOFT')

    def __call__(self, mesh):
        im = render_model(mesh, self.sgrid)
        return im

    def __repr__(self):
        return self.__class__.__name__ + '(bandwidth={0})'.format(self.bandwidth)
