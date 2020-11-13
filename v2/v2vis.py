import matplotlib
matplotlib.use('TKAGG')
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from mpl_toolkits.mplot3d import Axes3D, art3d  # noqa: F401 unused import


class V2Vis:
    def __init__(self, v2generator):
        self.v2generator = v2generator

    def plt_mesh(self):
        self.v2generator.mesh.show()

    def plt_convh(self):
        self.v2generator.convex_hull.show()

    def plt_v2_mesh_d(self):
        plt.imshow(self.v2generator.mesh_v2_d, vmin=0, vmax=2 * self.v2generator.r, cmap='gray')
        plt.axis('off')
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()
        plt.show()

    def plt_v2_repr(self):
        fig, axes = plt.subplots(2, 3)
        reprs = [self.v2generator.mesh_v2_d, self.v2generator.mesh_v2_s, self.v2generator.mesh_v2_c,
                 self.v2generator.convh_v2_d, self.v2generator.convh_v2_s, self.v2generator.convh_v2_c]

        for i, ax in enumerate(axes.ravel()):
            # 0 and 3 are depth channels ranging in [0 - 2r], else are sine/cosine channels raning in [0 - 1]
            ax.imshow(reprs[i], vmin=0, vmax=2 * self.v2generator.r if i in [0, 3] else 1, cmap='gray')
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.tight_layout()
        plt.show()

    def plt_v2_config(self, ax=None, mesh_p=True, convh_p=False, mesh=False, convh=False, **kwargs):
        """ Plot the object and the v2 camera configuration

        Args:
            ax: Axes
                Which ax to plot the v2 configuration
            mesh_p: bool
                Whether to plot the intersection points between the mesh and the rays or not
            convh_p:
                Whether to plot the intersection points between the convex hull and the rays or not
            mesh:
                Whether to directly plot the mesh by face triangles or not
            convh:
                Whether to directly plot the convex hull by face triangles or not
        """
        show_title = kwargs.get('show_title', False)
        scatter_size = kwargs.get('scatter_size', 20)
        title_size = kwargs.get('title_size', 30)
        plt_show = kwargs.get('plt_show', False)

        if ax is None:
            ax = Axes3D(plt.figure(figsize=(10, 10)))

        # Plot the ray origins
        cube_d = self.v2generator.r
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-cube_d, cube_d)
        ax.set_ylim(-cube_d, cube_d)
        ax.set_zlim(-cube_d, cube_d)
        ax.scatter(
            self.v2generator.ray_orig_np[:, 0],
            self.v2generator.ray_orig_np[:, 1],
            self.v2generator.ray_orig_np[:, 2],
            s=scatter_size
        )

        m, n, h, w, d = self.v2generator.m, self.v2generator.n, self.v2generator.h, self.v2generator.w, self.v2generator.d
        if show_title:
            ax.set_title(f'm={m} n={n}\nh={h} w={w} d={d:.2f}', size=title_size, y=0.92)

        # Plot the intersection points of the ray and the object mesh
        if mesh_p:
            mesh_v2_p = self.v2generator.mesh_v2_p[~(self.v2generator.mesh_v2_d == 2 * self.v2generator.r).flatten()]
            ax.scatter(mesh_v2_p[:, 0], mesh_v2_p[:, 1], mesh_v2_p[:, 2], s=scatter_size)

        # Plot the intersection points of the ray and the object convex hull
        if convh_p:
            convh_v2_p = self.v2generator.convh_v2_p[~(self.v2generator.convh_v2_d == 2 * self.v2generator.r).flatten()]
            ax.scatter(convh_v2_p[:, 0], convh_v2_p[:, 1], convh_v2_p[:, 2], s=scatter_size)

        # Plot the object mesh
        if mesh:
            for vtx in self.v2generator.mesh.triangles:
                tri = art3d.Poly3DCollection([vtx])
                tri.set_color(colors.rgb2hex(sp.rand(3)))
                tri.set_edgecolor('k')
                ax.add_collection3d(tri)

        # Plot the object convex hull
        if convh:
            for vtx in self.v2generator.convex_hull.triangles:
                tri = art3d.Poly3DCollection([vtx])
                tri.set_color(colors.rgb2hex(sp.rand(3)))
                tri.set_edgecolor('k')
                ax.add_collection3d(tri)

        ax.axis('off')

        if plt_show:
            plt.show()
