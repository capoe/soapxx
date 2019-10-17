# Adapted from https://github.com/antiprism/antiprism_python/anti_lib_progs

# Copyright (c) 2003-2016 Adrian Rossiter <adrian@antiprism.com>
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import numpy as np
import fractions

def get_octahedron(verts, faces):
    X = 0.25 * np.sqrt(2)
    verts.extend([GeoVec(0.0, 0.5, 0.0), GeoVec(X, 0.0, -X),
                  GeoVec(X, 0.0, X), GeoVec(-X, 0.0, X),
                  GeoVec(-X, 0.0, -X), GeoVec(0.0, -0.5, 0.0)])

    faces.extend([(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
                  (5, 2, 1), (2, 5, 3), (3, 5, 4), (4, 5, 1)])

def get_tetrahedron(verts, faces):
    X = 1 / np.sqrt(3)
    verts.extend([GeoVec(-X, X, -X), GeoVec(-X, -X, X),
                  GeoVec(X, X, X), GeoVec(X, -X, -X)])
    faces.extend([(0, 1, 2), (0, 3, 1), (0, 2, 3), (2, 1, 3)])

def get_ico_coords():
    phi = (np.sqrt(5) + 1) / 2
    rad = np.sqrt(phi+2)
    return 1/rad, phi/rad

def get_triangle(verts, faces):
    if 1:
        Y = np.sqrt(3.0) / 12.0
        Z = -0.8
        verts.extend([GeoVec(-0.25, -Y, Z), GeoVec(0.25, -Y, Z),
                      GeoVec(0.0, 2 * Y, Z)])
        faces.extend([(0, 1, 2)])
    else:
        X, Z = get_ico_coords()
        verts.extend([GeoVec(-X, 0.0, -Z), GeoVec(X, 0.0, -Z),
                      GeoVec(0.0, Z, -X), GeoVec(0.0, -Z, -X)])
        faces.extend([(0, 1, 2), (0, 3, 1)])

def get_icosahedron(verts, faces):
    X, Z = get_ico_coords()
    verts.extend([GeoVec(-X, 0.0, Z), GeoVec(X, 0.0, Z), GeoVec(-X, 0.0, -Z),
                  GeoVec(X, 0.0, -Z), GeoVec(0.0, Z, X), GeoVec(0.0, Z, -X),
                  GeoVec(0.0, -Z, X), GeoVec(0.0, -Z, -X), GeoVec(Z, X, 0.0),
                  GeoVec(-Z, X, 0.0), GeoVec(Z, -X, 0.0), GeoVec(-Z, -X, 0.0)])
    faces.extend([(0, 4, 1), (0, 9, 4), (9, 5, 4), (4, 5, 8), (4, 8, 1),
                  (8, 10, 1), (8, 3, 10), (5, 3, 8), (5, 2, 3), (2, 7, 3),
                  (7, 10, 3), (7, 6, 10), (7, 11, 6), (11, 0, 6), (0, 1, 6),
                  (6, 1, 10), (9, 0, 11), (9, 11, 2), (9, 2, 5), (7, 2, 11)])

def get_poly(poly, verts, edges, faces):
    if poly == 'i':
        get_icosahedron(verts, faces)
    elif poly == 'o':
        get_octahedron(verts, faces)
    elif poly == 't':
        get_tetrahedron(verts, faces)
    elif poly == 'T':
        get_triangle(verts, faces)
    else:
        return 0
    for face in faces:
        for i in range(0, len(face)):
            i2 = i + 1
            if(i2 == len(face)):
                i2 = 0

            if face[i] < face[i2]:
                edges[(face[i], face[i2])] = 0
            else:
                edges[(face[i2], face[i])] = 0
    return 1

def grid_to_points(grid, freq, div_by_len, f_verts, face):
    """Convert grid coordinates to Cartesian coordinates"""
    points = []
    v = []
    for vtx in range(3):
        v.append([GeoVec(0.0, 0.0, 0.0)])
        edge_vec = f_verts[(vtx + 1) % 3] - f_verts[vtx]
        if div_by_len:
            for i in range(1, freq + 1):
                v[vtx].append(edge_vec * float(i) / freq)
        else:
            ang = 2 * np.arcsin(edge_vec.mag() / 2.0)
            unit_edge_vec = edge_vec.unit()
            for i in range(1, freq + 1):
                len = np.sin(i * ang / freq) / \
                    np.sin(np.pi / 2 + ang / 2 - i * ang / freq)
                v[vtx].append(unit_edge_vec * len)
    for (i, j) in grid.values():
        if (i == 0) + (j == 0) + (i + j == freq) == 2:   # skip vertex
            continue
        # skip edges in one direction
        if (i == 0 and face[2] > face[0]) or (
                j == 0 and face[0] > face[1]) or (
                i + j == freq and face[1] > face[2]):
            continue
        n = [i, j, freq - i - j]
        v_delta = (v[0][n[0]] + v[(0-1) % 3][freq - n[(0+1) % 3]] -
                   v[(0-1) % 3][freq])
        pt = f_verts[0] + v_delta
        if not div_by_len:
            for k in [1, 2]:
                v_delta = (v[k][n[k]] + v[(k-1) % 3][freq - n[(k+1) % 3]] -
                           v[(k-1) % 3][freq])
                pt = pt + f_verts[k] + v_delta
            pt = pt / 3
        points.append(pt)
    return points

def make_grid(freq, m, n):
    """Make the geodesic pattern grid"""
    grid = {}
    rng = (2 * freq) // (m + n)
    for i in range(rng):
        for j in range(rng):
            x = i * (-n) + j * (m + n)
            y = i * (m + n) + j * (-m)
            if x >= 0 and y >= 0 and x + y <= freq:
                grid[(i, j)] = (x, y)
    return grid

def class_type(val_str):
    """Read the class pattern specifier"""
    order = ['first', 'second']
    num_parts = val_str.count(',')+1
    vals = val_str.split(',', 2)
    if num_parts == 1:
        if vals[0] == '1':
            pat = [1, 0, 1]
        elif vals[0] == '2':
            pat = [1, 1, 1]
        else:
            raise argparse.ArgumentTypeError(
                'class type can only be 1 or 2 when a single value is given')
    elif num_parts == 2:
        pat = []
        for i, num_str in enumerate(vals):
            try:
                num = int(num_str)
            except:
                raise argparse.ArgumentTypeError(
                    order[i] + ' class pattern value not an integer')
            if num < 0:
                raise argparse.ArgumentTypeError(
                    order[i] + ' class pattern cannot be negative')
            if num == 0 and i == 1 and pat[0] == 0:
                raise argparse.ArgumentTypeError(
                    ' class pattern values cannot both be 0')
            pat.append(num)
        rep = fractions.gcd(*pat)
        pat = [pat_num//rep for pat_num in pat]
        pat.append(rep)
    else:
        raise argparse.ArgumentTypeError(
            'class type contains more than two values')
    return pat

class GeoVec:
    def __init__(self, *v):
        self.v = list(v)
    def to_numpy(self):
        return np.array(self.v)
    def fromlist(self, v):
        if not isinstance(v, list):
            raise TypeError
        self.v = v[:]
        return self
    def copy(self):
        return GeoVec().fromlist(self.v)
    def __str__(self):
        return '(' + repr(self.v)[1:-1] + ')'
    def __repr__(self):
        return 'GeoVec(' + repr(self.v)[1:-1] + ')'
    def __len__(self):
        return len(self.v)
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.v):
            raise KeyError
        return self.v[key]
    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.v):
            raise KeyError
        self.v[key] = value
    def __neg__(self):
        v = list(map(lambda x: -x, self.v))
        return GeoVec().fromlist(v)
    def __add__(self, other):
        v = list(map(lambda x, y: x+y, self.v, other.v))
        return GeoVec().fromlist(v)
    def __sub__(self, other):
        v = list(map(lambda x, y: x-y, self.v, other.v))
        return GeoVec().fromlist(v)
    def __mul__(self, scalar):
        v = list(map(lambda x: x*scalar, self.v))
        return GeoVec().fromlist(v)
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    def __truediv__(self, scalar):
        return self.__mul__(1/scalar)
    def __div__(self, scalar):
        return self.__mul__(1./scalar)
    def mag2(self):
        return self.dot(self, self)
    def mag(self):
        return np.sqrt(self.mag2())
    def unit(self):
        return self.__truediv__(self.mag())
    def rot_z(self, ang):
        r = np.sqrt(self.v[0]**2+self.v[1]**2)
        initial_ang = np.atan2(self.v[1], self.v[0])
        final_ang = initial_ang + ang
        return GeoVec(r*np.cos(final_ang), r*np.sin(final_ang), self.v[2])
    @staticmethod
    def cross(v0, v1):
        return GeoVec(v1[2]*v0[1] - v1[1]*v0[2],
                   v1[0]*v0[2] - v1[2]*v0[0],
                   v1[1]*v0[0] - v1[0]*v0[1])
    @staticmethod
    def dot(v0, v1):
        return sum(map(lambda x, y: x*y, v0.v, v1.v))
    @staticmethod
    def triple(v0, v1, v2):
        return GeoVec.dot(v0, GeoVec.cross(v1, v2))

def pattern_to_numpy(points):
    grid = np.zeros((len(points),3))
    for i, p in enumerate(points):
        grid[i] = p.to_numpy()
    return grid

def subdivide_sphere(
        args_polyhedron = "i",
        args_class_pattern = [1,0,1],
        args_repeats = 10,
        args_radius = 4.5,
        args_equal_length = False,
        args_flat_faced = False):
    # >>> parser = argparse.ArgumentParser(formatter_class=anti_lib.DefFormatter,
    # >>>                                  description=__doc__, epilog=epilog)
    # >>> parser.add_argument(
    # >>>     'repeats',
    # >>>     help='number of times the pattern is repeated (default: 1)',
    # >>>     type=anti_lib.read_positive_int,
    # >>>     nargs='?',
    # >>>     default=1)
    # >>> parser.add_argument(
    # >>>     '-p', '--polyhedron',
    # >>>     help='base polyhedron: i - icosahedron (default), '
    # >>>          'o - octahedron, t - tetrahedron, T - triangle.',
    # >>>     choices=['i', 'o', 't', 'T'],
    # >>>     default='i')
    # >>> parser.add_argument(
    # >>>     '-c', '--class-pattern',
    # >>>     help='class of face division,  1 (Class I, default) or '
    # >>>          '2 (Class II), or two numbers separated by a comma to '
    # >>>          'determine the pattern (Class III generally, but 1,0 is '
    # >>>          'Class I, 1,1 is Class II, etc).',
    # >>>     type=class_type,
    # >>>     default=[1, 0, 1])
    # >>> parser.add_argument(
    # >>>     '-f', '--flat-faced',
    # >>>     help='keep flat-faced polyhedron rather than projecting '
    # >>>          'the points onto a sphere.',
    # >>>     action='store_true')
    # >>> parser.add_argument(
    # >>>     '-l', '--equal-length',
    # >>>     help='divide the edges by equal lengths rather than equal angles',
    # >>>     action='store_true')
    # >>> parser.add_argument(
    # >>>     '-o', '--outfile',
    # >>>     help='output file name (default: standard output)',
    # >>>     type=argparse.FileType('w'),
    # >>>     default=sys.stdout)
    # >>> args = parser.parse_args()
    verts = []
    edges = {}
    faces = []
    get_poly(args_polyhedron, verts, edges, faces)
    (M, N, reps) = args_class_pattern
    repeats = args_repeats * reps
    freq = repeats * (M**2 + M*N + N**2)
    grid = {}
    grid = make_grid(freq, M, N)
    points = verts
    for face in faces:
        if args_polyhedron == 'T':
            face_edges = (0, 0, 0)  # generate points for all edges
        else:
            face_edges = face
        points[len(points):len(points)] = grid_to_points(
            grid, freq, args_equal_length,
            [verts[face[i]] for i in range(3)], face_edges)
    if not args_flat_faced:
        points = [args_radius*p.unit() for p in points]  # Project onto sphere
    return pattern_to_numpy(points)

if __name__ == "__main__":
    grid_all = []
    repeats = [ 6, 7, 8, 9, 10 ]
    for i, r in enumerate([ 2.5, 3.0, 3.5, 4.0, 4.5 ]):
        S = 4*np.pi*r**2
        N = S/0.5**2
        grid = subdivide_sphere(args_radius=r, args_repeats=repeats[i])
        grid_all.append(grid)
    grid = np.concatenate(grid_all, axis=0)

