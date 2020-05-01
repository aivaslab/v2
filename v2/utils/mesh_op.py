import numpy as np
import trimesh


def rotmat(z_, y_, x_, hom_coord):
    """ Construct rotation matrix to rotate mesh

    Args:
        z_: int
            rotation degree around z-axis
        y_: int
            rotation degree around y-axis
        x_: int
            rotation degree around x-axis
        hom_coord:

    Returns:

    """

    def z(a):
        return np.array([[np.cos(a), -np.sin(a), 0, 0],
                         [np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, np.sin(a), 0],
                         [0, 1, 0, 0],
                         [-np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    def x(a):
        return np.array([[1, 0, 0, 0],
                         [0, np.cos(a), -np.sin(a), 0],
                         [0, np.sin(a), np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(z_).dot(y(y_)).dot(x(x_))
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, hom_coord=True)
    return rot


def fix_rot(rot_zyx):
    z, y, x = rot_zyx
    rot = rotmat(z, y, x, hom_coord=True)
    return rot


class ToMesh:
    def __init__(self, rot, tr):
        self.rot = rot
        self.tr = tr

    def __call__(self, path):
        mesh = trimesh.load_mesh(path)
        mesh.remove_degenerate_faces()
        mesh.fix_normals()
        mesh.fill_holes()
        # mesh.remove_duplicate_faces()  # Will remove faces that are actually not duplicated
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        mesh.apply_translation(-mesh.centroid)

        r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
        mesh.apply_scale(1 / r)

        mesh = self.apply_tr(mesh)
        mesh = self.apply_rot(mesh)

        r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
        mesh.apply_scale(0.99 / r)

        return mesh

    def apply_tr(self, mesh):
        rot = rnd_rot()
        mesh.apply_transform(rot)
        mesh.apply_translation([self.tr, 0, 0])
        mesh.apply_transform(rot.T)
        return mesh

    def apply_rot(self, mesh):
        mesh.apply_transform(fix_rot(self.rot))
        return mesh

    def __repr__(self):
        return self.__class__.__name__ + '(rotation={}, translation={})'.format(self.rot, self.tr)
