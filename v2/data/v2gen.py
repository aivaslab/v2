import os
import math
import time
import itertools

from v2.data.v2lib import V2Lib
from v2.utils import conf, util


class V2Generator(V2Lib):
    def __init__(self, m, n, w, h, d, r, polar, ssm, v2m, rot, tr):
        """ V2 Generator

        Args:
            m: int
                Rows in view (e.g. latitude lines)

            n: int
                Columns in view (e.g. longitude lines)

            w: int
                Height of each view, in pixels

            h: int
                Width of each view, in pixels

            d: float
                The distance between two nearest view points

            r: float
                The radius of the ray sphere

            polar:
                Include the polar point or not

            ssm: str, ['uniform','s2cnn_dh', 'trimesh_uv']
                Sphere sampling method
                'uniform': uniformly distribute sampling points on the sphere to alleviate polar effect
                's2cnn_dh': A sampling method in s2cnn. 'dh' is Driscoll-Healy grid
                'trimesh_uv': UV sphere sampling in trimesh library

            v2m: str, ['trimesh', 'e2f', 'mt', 's2cnn']
                V2 generating method
                'trimesh': RayMeshIntersector class intersects_location() method in trimesh library
                'e2f': Naive ray-triangle intersection implemented from scratch by Tengyu
                    (well... some corner cases of sine and cosine may not be correctly implemented)
                'mt': Möller–Trumbore intersection algorithm, the fastest method
                    (Vectorized by Tengyu)
                's2cnn': Ray-triangle intersection implemented in s2cnn paper. In fact, it is the same as the 'trimesh'
                    method, as it calculates the intersection points by the same method in trimesh library

            rot: list, [z, y, x]
                Fix rotation, specify by rotations on a, z, c axis
                z: z-axis in Toybox, z+ direction
                y: y-axis in Toybox, y+ direction
                x: x-axis in Toybox, x+ direction

            tr: float
                Translation value, add translation to the loaded mesh

        """
        super().__init__(m, n, w, h, d, r, polar, ssm)
        self.v2m = v2m
        self.rot = rot
        self.tr = tr

    def __call__(self, mesh_path):
        """ Load mesh file and add data augmentation

        Args:
            mesh_path: str
                the file path for the .obj mesh file

        """
        super().load_mesh(mesh_path, self.rot, self.tr)
        super().v2repr(self.v2m)


def main():
    # V2 subdivisions based on the perfect factors of total pixels
    S = 128  # Side Pixels
    C = S ** 2  # Total Pixels
    facs = util.get_perfect_factors(C)

    # Rotation matrix around z-, y-, and x- axis, every axis has 12 rotations
    rot_z = list(range(12)) + [0] * 24
    rot_y = [0] * 12 + list(range(12)) + [0] * 12
    rot_x = [0] * 24 + list(range(12))
    rot_zyx = list(zip(rot_z, rot_y, rot_x))
    rot_zyx = rot_zyx[: 12] + rot_zyx[13: 24] + rot_zyx[25:]

    # Get all mesh files
    # 3193 is an airplane mesh with small number of faces. Good for debugging
    train_set, test_set = util.modelnet40_aligned()
    mesh_paths = train_set + test_set

    # All configurations
    all_config = [rot_zyx, facs, mesh_paths]
    all_config = list(itertools.product(*all_config))

    last_conf_i = 175268
    last_zyx, last_fac, last_obj = all_config[last_conf_i]
    last_rot_i = rot_zyx.index(last_zyx)
    last_fac_i = facs.index(last_fac)
    last_mesh_i = mesh_paths.index(last_obj)

    start = time.time()

    for conf_i, all_config_comb in enumerate(all_config[last_conf_i:]):

        zyx, fac, obj = all_config_comb
        rot_i = rot_zyx.index(zyx)
        fac_i = facs.index(fac)
        mesh_i = mesh_paths.index(obj)

        z, y, x = zyx

        m = n = int(math.sqrt(fac))
        h = w = int(math.sqrt(C // fac))
        d = 2 / S
        r = 1
        polar = True
        ssm = 'v2_soft'
        v2m = 'mt'
        tr = 0
        rot = [z / 12 * 2 * math.pi, y / 12 * 2 * math.pi, x / 12 * 2 * math.pi]
        v2_config = '{}_{}_{}_{}_{}|{}'.format(m, n, h, w, 2, S)

        v2generator = V2Generator(m, n, w, h, d, r, polar, ssm, v2m, rot, tr)
        v2generator(obj)

        ca_, set_, id_ = util.get_ca_set_id(obj)
        dst = os.path.join(conf.ModelNet40_DSCDSC,
                           '{}_C{}'.format('SOFT', C),
                           ca_,
                           set_,
                           id_,
                           '{}'.format('_'.join(map(lambda x_: '{:.0f}'.format(x_), zyx))),
                           '{}.pickle'.format(v2_config)
                           )

        dir_ = os.path.dirname(dst)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        v2generator.save_v2(dst)

        logger_content = 'All: {}/{}, Rotation: {}/{} {}, V2 Configuration: {}/{}, {}' \
                         ' Object No: {}/{} {}, Time Spent: {}/{}'.format(
            conf_i + 1 + last_conf_i, len(all_config),
            rot_i + 1, len(rot_zyx), '{}'.format('_'.join(map(lambda x_: '{:.0f}'.format(x_), zyx))),
            fac_i + 1, len(facs), v2_config,
            mesh_i + 1, len(mesh_paths), os.path.basename(obj),
            (time.time() - start), (time.time() - start) / (conf_i + 1) * (len(all_config) - last_conf_i))

        print(logger_content)
        with open('logger.txt', 'a+') as f:
            f.write(logger_content)
            f.write('\n')


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('main()', sort='tottime')
    main()
