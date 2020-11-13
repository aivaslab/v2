import math
import matplotlib.pyplot as plt

from v2.v2gen import V2Generator
from v2.v2vis import V2Vis


def main():
    m = 32  # view row
    n = 32  # view col
    h = 1  # height per view
    w = 1  # width per view
    d = 2 / 32  # distance between two closest pixels
    r = 1  # sphere radius
    polar = True  # sampling viewpoint on polar or not
    ssm = 'v2_soft'  # sphere sampling method
    v2m = 'mt'  # v2 generating method
    tr = [0, 0, 0]  # List[x+, y+, z+] translation distance
    rot = [0 / 12 * 2 * math.pi, 0 / 12 * 2 * math.pi, 0 / 12 * 2 * math.pi]  # List[z+, y+, x+] euler angles

    v2generator = V2Generator(m, n, w, h, d, r, polar, ssm, v2m, rot, tr)

    off_file = 'mesh/airplane_0001.off'
    v2generator(off_file)

    v2visualizer = V2Vis(v2generator)

    v2visualizer.plt_mesh()
    v2visualizer.plt_v2_repr()
    v2visualizer.plt_v2_config(ax=None, mesh_p=True, convh_p=False, mesh=True, convh=False, plt_show=True)


if __name__ == '__main__':
    main()
