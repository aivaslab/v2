import os
import torch
import glob
import matplotlib
# matplotlib.use('TKAGG')
import matplotlib.pyplot as plt
import trimesh
import numpy as np

from trimesh.ray.ray_triangle import RayMeshIntersector
from trimesh import creation
from mpl_toolkits.mplot3d import Axes3D
from v2.util import conf


class V2Generator:
    def __init__(self, m, n, w, h, d, r, polar):
        self.m = m  # Rows in view (e.g. latitude lines)
        self.n = n  # Columns in view (e.g. longitude lines)
        self.h = h  # Height of each view, in pixels
        self.w = w  # Width of each view, in pixels
        self.d = d  # The distance between two nearest view points
        self.r = r
        self.polar = polar  # Include the polar point or not
        self.s_view = self.uv_sphere()
        # self.s_view = self.trimesh_uv_sphere()
        self.ray_origins, self.ray_edges, self.ray_directions = self.get_rays()
        
        self.obj_file = None
        self.mesh = None
        self.convex_hull = None

        self.convex_hull_v2_d = self.mesh_v2_d = None  # Ray travelling distance, range [0, 1] as it is inside a unit sphere
        self.convex_hull_v2_a = self.mesh_v2_a = None  # Incident angle, range [0, pi/2]
        self.convex_hull_v2_s = self.mesh_v2_s = None  # Sine of incident angle, range [0, 1]
        self.convex_hull_v2_c = self.mesh_v2_c = None  # Cosine of incident angle, range [0, 1]

    def get_rays(self):
        """ A function to generate views such that:
          V2 representations are generated by normalizing a given 3D object inside a sphere, and sampling points on
          the sphere to form a view plane tangent to the sphere. Then, a view plane will shoot rays parallel to the
           normal towards the object, finally reaching the object (or missing it, becoming a “background” pixel).

        Returns:
            ray_origins: ndarray, (m*n*x*y, 3)
                Origin points to shoot rays.
            ray_edges: ndarray, (m*n*x*y, 2, 3)
                Ray segments.
            ray_directions: ndarray, (m*n*x*y, 3)
                Ray directions starting at origins
        """

        s_d = -2 * self.s_view  # the direction vector shooting to the (0, 0, 0) with length 1, -self.s_view - self.s_view

        # The orthonormal basis on the view surface
        x0 = self.s_view[0, :]
        y0 = self.s_view[1, :]
        z0 = self.s_view[2, :]

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

        vz = self.s_view  # z basis, the one shooting to the origin
        vz = vz / np.linalg.norm(vz, 2, axis=0)

        # The meshgrid for the new orthonormal basis
        dx = np.linspace(-(self.w // 2) * self.d, self.w // 2 * self.d, self.w)
        dy = np.linspace(-(self.h // 2) * self.d, self.h // 2 * self.d, self.h)
        dx, dy = np.meshgrid(dx, dy)
        dx = dx.flatten()
        dy = dy.flatten()

        dvx = np.tile(vx, (len(dx), 1, 1)).transpose((1, 2, 0)) * dx
        dvy = np.tile(vy, (len(dy), 1, 1)).transpose((1, 2, 0)) * dy

        # Generate all view
        all_view = np.tile(self.s_view, (len(dx), 1, 1)).transpose((1, 2, 0)) + dvx + dvy
        all_ray = all_view + np.tile(s_d, (len(dx), 1, 1)).transpose((1, 2, 0))

        all_view = all_view.reshape((all_view.shape[0], all_view.shape[1] // self.n, self.n, self.h, self.w))
        all_view = all_view.transpose(0, 1, 3, 2, 4)
        all_ray = all_ray.reshape((all_ray.shape[0], all_ray.shape[1] // self.n, self.n, self.h, self.w))
        all_ray = all_ray.transpose(0, 1, 3, 2, 4)

        all_view = all_view.reshape((3, -1))
        all_ray = all_ray.reshape((3, -1))

        all_ray = np.dstack((all_view, all_ray)).transpose((1, 2, 0))

        ray_origins = all_view.T
        ray_edges = all_ray
        ray_directions = all_ray[:, 1, :] - all_ray[:, 0, :]
        return ray_origins, ray_edges, ray_directions

    def uv_sphere(self):
        """ Generate points on a sphere based on the number of sampled points and the radius of the sphere.
        Ref: http://corysimon.github.io/articles/uniformdistn-on-sphere/
        """

        theta = np.linspace(0, 1, self.n, endpoint=False)
        theta = 2 * np.pi * theta

        if not self.polar:
            phi = np.linspace(0, 1, self.m + 2)  # +2 because we will remove both polar points later
            phi = phi[~np.logical_or(phi == 0, phi == 1)]
        else:
            phi = np.linspace(0, 1, self.m)
        phi = np.arccos(1 - 2 * phi)

        theta, phi = np.meshgrid(theta, phi)
        theta = theta.flatten()
        phi = phi.flatten()

        # When phi is 0 or pi, the point is on the z-axis, and therefore x, y should be zero
        theta[np.logical_or(phi == 0, phi == np.pi)] = 0

        x = self.r * self._sin(phi) * self._cos(theta)
        y = self.r * self._sin(phi) * self._sin(theta)
        z = self.r * self._cos(phi)
        s_view = np.vstack((x, y, z))
        return s_view

    def trimesh_uv_sphere(self):
        return np.array(creation.uv_sphere(radius=self.r, count=[self.m, (self.n + 1) // 2]).vertices).T

    def load_obj(self, obj_file):
        self.obj_file = obj_file
        
        loaded_obj = trimesh.exchange.obj.load_obj(open(obj_file, 'r'))
        verts = loaded_obj['vertices']
        faces = loaded_obj['faces']

        center = verts.mean(0)
        verts = verts - center
        scale = np.max(np.abs(verts)) * 2  # Scale to fit a sphere of diameter 1
        verts = verts / scale

        self.mesh = mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        self.convex_hull = convex_hull = self.mesh.convex_hull
        return mesh, convex_hull
    
    def v2repr(self):
        self.mesh_v2_d, self.mesh_v2_a, self.mesh_v2_s, self.mesh_v2_c = self.v2repr_core(self.mesh)
        self.convex_hull_v2_d, self.convex_hull_v2_a, self.convex_hull_v2_s, self.convex_hull_v2_c = self.v2repr_core(self.convex_hull)

    def v2repr_core(self, mesh):
        """ Trimesh mesh to generate v2 representation

        Args:
            mesh: (Trimesh)

        Returns:

        """
        inter_points, index_ray, index_tri = self.trimesh_intersection(mesh)
        normals = mesh.face_normals[index_tri]
        ray_directions = self.ray_directions[index_ray]

        v2_d = np.full(len(self.ray_origins), 1.0)
        v2_a = np.zeros(len(self.ray_origins))

        d_ = np.linalg.norm(self.ray_origins[index_ray] - inter_points, axis=1)
        a_ = self._angle(ray_directions, normals)
        a_[a_ > np.pi / 2] = np.pi - a_[a_ > np.pi / 2]

        v2_d[index_ray] = d_
        v2_a[index_ray] = a_
        v2_s = self._sin(v2_a)
        v2_c = self._cos(v2_a)

        v2_d = v2_d.reshape((self.m, self.n))
        v2_a = v2_a.reshape((self.m, self.n))
        v2_s = v2_s.reshape((self.m, self.n))
        v2_c = v2_c.reshape((self.m, self.n))
        
        return v2_d, v2_a, v2_s, v2_c 

    def trimesh_intersection(self, mesh):
        intersector = RayMeshIntersector(mesh)
        return intersector.intersects_location(self.ray_origins, self.ray_directions, multiple_hits=False)

    def plt_v2_repr(self):
        fig, axes = plt.subplots(2, 3)
        reprs = [self.mesh_v2_d, self.mesh_v2_s, self.mesh_v2_c,
                 self.convex_hull_v2_d, self.convex_hull_v2_s, self.convex_hull_v2_c]
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(reprs[i], vmin=0, vmax=1, cmap='gray')
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.tight_layout()
        plt.show()

    def plt_v2_config(self):
        mesh_inter_points, _, _ = self.trimesh_intersection(self.mesh)
        convex_hull_inter_points, _, _ = self.trimesh_intersection(self.convex_hull)

        ax = Axes3D(plt.figure(figsize=(10, 10)))
        cube_d = 0.5
        ax.set_xlim(-cube_d, cube_d)
        ax.set_ylim(-cube_d, cube_d)
        ax.set_zlim(-cube_d, cube_d)
        ax.scatter(self.ray_origins[:, 0], self.ray_origins[:, 1], self.ray_origins[:, 2])
        ax.scatter(mesh_inter_points[:, 0], mesh_inter_points[:, 1], mesh_inter_points[:, 2])
        ax.scatter(convex_hull_inter_points[:, 0], convex_hull_inter_points[:, 1], convex_hull_inter_points[:, 2])

        plt.axis('off')
        plt.show()

    def _angle(self, v1, v2):
        """ Returns the angle in radians between vectors v1 and v2
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))

    @staticmethod
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector, axis=1, keepdims=True)

    @staticmethod
    def _sin(x):
        # To correct np.sin(np.pi) isn't equal to 0
        sin = np.sin(x)
        sin[np.abs(sin) < 1e-14] = 0
        return sin

    @staticmethod
    def _cos(x):
        # To correct np.cos(np.pi/2) isn't equal to 0
        cos = np.cos(x)
        cos[np.abs(cos) < 1e-14] = 0
        return cos


def main():
    objs = sorted(glob.glob(conf.ModelNet40OBJ_DIR + '/*/*/*.obj'))
    obj_file = objs[0]
    
    m = 32
    n = 32
    w = 1
    h = 1
    d = 1 / 32
    r = 0.5
    polar = False
    
    v2generator = V2Generator(m, n, w, h, d, r, polar)
    v2generator.load_obj(obj_file)

    # v2generator.v2repr()
    #
    # v2generator.plt_v2_config()
    # v2generator.plt_v2_repr()

    v2generator.mesh.show()
    v2generator.convex_hull.show()


if __name__ == '__main__':
    main()
