from v2.v2lib import V2Lib


class V2Generator(V2Lib):
    def __init__(self, m, n, w, h, d, r, polar, ssm, v2m, rot, tr):
        """ V2 Generator

        Args:
            m: int
                Rows in view (e.g. latitude lines)
            n: int
                Columns in view (e.g. longitude lines)
            w: int
                Width of each view, in pixels
            h: int
                Height of each view, in pixels
            d: float
                The distance between two nearest view points
            r: float
                The radius of the ray sphere
            polar:
                Include the polar point or not
            ssm: str, ['v2_uni', 'v2_dh', 'v2_soft', 'v2_fibonacci', 's2cnn_dh', 's2cnn_soft', 'trimesh_uv']
                sphere sampling method:
                v2_ series are the sampling methods implemented internally with v2 representation
                s2cnn_ series are the sampling methods directly adopted from the s2cnn repo
                trimesh_ series are the uv unwrapping methods from the trimesh module
            v2m: str, ['trimesh', 'e2f', 'mt', 's2cnn']
                V2 generating method
                'trimesh': RayMeshIntersector class intersects_location() method in trimesh library
                'e2f': Naive ray-triangle intersection implemented from scratch
                    (well... some corner cases of sine and cosine may not be correctly implemented)
                'mt': Möller–Trumbore intersection algorithm, the fastest method
                's2cnn': Ray-triangle intersection implemented in s2cnn paper. In fact, it is the same as the 'trimesh'
                    method, as it calculates the intersection points by the same method in trimesh library
            rot: list, [z+, y+, x+]
                Rotation around z+, y+, x+ axis
            tr: list, [x+, y+, z+]
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
