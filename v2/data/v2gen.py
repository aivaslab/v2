import os
import math
from v2.data.v2lib import V2Lib
from v2.utils import conf, util


class V2Generator(V2Lib):
    def __init__(self, m, n, w, h, d, r, polar, ssm, v2m, rrot, frot, rtr):
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

            rrot: bool
                Random rotation, whether to add random rotation to the loaded mesh or not

            frot: list, [z, y, x]
                Fix rotation, specify by rotations on a, z, c axis
                z: z-axis in Toybox, z+ direction
                y: y-axis in Toybox, y+ direction
                x: x-axis in Toybox, x+ direction

            rtr: float
                Random translation, how large to add random translation to the loaded mesh

        """
        super().__init__(m, n, w, h, d, r, polar, ssm)
        self.v2m = v2m
        self.rrot = rrot
        self.frot = frot
        self.rtr = rtr

    def __call__(self, obj_file):
        """ Load mesh file and add data augmentation

        Args:
            obj_file: str
                the file path for the .obj mesh file

        """
        super().load_obj(obj_file, self.rrot, self.frot, self.rtr)
        super().v2repr(self.v2m)


def main():
    m = n = 15
    h = w = 5
    d = 1 / 20
    r = 1
    polar = True
    ssm = 'v2_soft'
    v2m = 'mt'
    rtr = 0
    rrot = False
    frot = [0 / 12 * 2 * math.pi, 0 / 12 * 2 * math.pi, 0 / 12 * 2 * math.pi]
    v2_config = '{}_{}_{}_{}_{:.4f}.npy'.format(m, n, h, w, d)

    v2generator = V2Generator(m, n, w, h, d, r, polar, ssm, v2m, rrot, frot, rtr)
    train_set, test_set = util.modelnet40_objs()
    objs = train_set + test_set

    import time
    start = time.time()

    # 3193 is an airplane mesh with small number of faces. Good for debugging
    last_i = 3193
    for i, obj in enumerate(objs[last_i:]):
        v2generator(obj)

        # old = conf.ModelNet40OBJ_DIR
        # new = os.path.join(conf.ModelNet40_MDSC_CDSC, 'ssm_{}_C{}'.format(ssm, m * h * n * w))
        # dst = util.get_dst(obj, old, new, v2_config)
        # v2generator.save_v2(dst)

        # Visualizations
        v2generator.plt_v2_config(mesh=True)
        v2generator.plt_v2_repr()
        # v2generator.plt_v2_mesh_d()
        # imgs.append(v2generator.mesh_v2_d)
        # v2generator.mesh.show()
        # v2generator.convex_hull.show()
        print('{}/{}, {}/{}'.format((time.time() - start), (time.time() - start) / (i + 1) * (len(objs) - last_i),
                                    i + 1, (len(objs) - last_i)))

        break


if __name__ == '__main__':
    import cProfile
    main()
    # cProfile.run('main()', sort='tottime')
