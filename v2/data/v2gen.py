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
    S = 128  # Side Pixels
    C = S ** 2  # Total Pixels
    facs = util.get_perfect_factors(C)
    frot_z = list(range(12)) + [0] * 24
    frot_y = [0] * 12 + list(range(12)) + [0] * 12
    frot_x = [0] * 24 + list(range(12))
    frot_i = list(zip(frot_z, frot_y, frot_x))
    frot_i = frot_i[1: 12] + frot_i[13: 24] + frot_i[25:]
    from v2.utils.vis import plt_v2_config, plt_v2_repr
    import matplotlib.pyplot as plt

    for z, y, x in frot_i[:]:
        # fig = plt.figure(figsize=(80, 80))
        # ax_i = 1

        for fac in facs:
            m = n = int(math.sqrt(fac))
            h = w = int(math.sqrt(C // fac))
            d = 2 / S
            r = 1
            polar = True
            ssm = 'v2_soft'
            v2m = 'mt'
            rtr = 0
            rrot = False
            frot = [z / 12 * 2 * math.pi, y / 12 * 2 * math.pi, x / 12 * 2 * math.pi]
            v2_config = '{}_{}_{}_{}_{}|{}'.format(m, n, h, w, 2, S)

            v2generator = V2Generator(m, n, w, h, d, r, polar, ssm, v2m, rrot, frot, rtr)
            train_set, test_set = util.modelnet40_objs()
            objs = train_set + test_set

            import time
            start = time.time()

            # 3193 is an airplane mesh with small number of faces. Good for debugging
            last_i = 0
            for i, obj in enumerate(objs[last_i:]):
                v2generator(obj)

                ca_, set_, id_ = util.get_ca_set_id(obj)
                dst = os.path.join(conf.ModelNet40_DSCDSC,
                                   '{}_C{}'.format('SOFT', C),
                                   ca_,
                                   set_,
                                   id_,
                                   '{}'.format('_'.join(map(lambda x: '{:.0f}'.format(x), frot))),
                                   '{}.pickle'.format(v2_config)
                                   )
                dir_ = os.path.dirname(dst)
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
                v2generator.save_v2(dst)

                # Visualizations
                # v2generator.plt_v2_config(mesh=True)
                # v2generator.plt_v2_repr()
                # v2generator.plt_v2_mesh_d()
                # imgs.append(v2generator.mesh_v2_d)
                # v2generator.mesh.show()
                # v2generator.convex_hull.show()

                # ax_j = ax_i
                #
                # ax = fig.add_subplot(8, 8, ax_j, projection='3d')
                # plt_v2_config(v2generator, convh=False, mesh=True, ax=ax)
                #
                # ax_j += 8
                # ax = fig.add_subplot(8, 8, ax_j, projection='3d')
                # plt_v2_config(v2generator, convh=True, mesh=True, ax=ax)
                #
                # for channel in range(6):
                #     ax_j += 8
                #     ax = fig.add_subplot(8, 8, ax_j)
                #     plt_v2_repr(v2generator, channel, ax)
                #
                # ax_i += 1
                print('{}/{}, {}/{}'.format((time.time() - start), (time.time() - start) / (i + 1) * (len(objs) - last_i),
                                            i + 1, (len(objs) - last_i)))
                # break

        # plt.savefig(os.path.join('/home/tengyu/Dropbox/meeting/04102020/v2_samples_C16384/{}_{}_{}.png'.format(z, y, x)))


if __name__ == '__main__':
    import cProfile
    main()
    # cProfile.run('main()', sort='tottime')
