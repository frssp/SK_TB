import numpy as np
import itertools

class Structure(object):
    """ Defines atomic structure"""
    def __init__(self, lattice, atoms, bond_length=2.7, periodicity=None, name=None):
        """Args:
            lattice:
                Lattice object
            atoms:
                list of Atom object
        """
        assert isinstance(lattice, Lattice), 'not Lattice object'
        assert isinstance(atoms, list), 'atoms is not list'
        assert isinstance(atoms[0], Atom), 'atom is not Atom object'
        
        self.name = name or 'system'
        self.lattice = lattice
        self.atoms = atoms
        self.bond_length = bond_length
        self.periodicity = periodicity or [True, True, True]

    def get_bond_mat(self):
        max_image = 3 ** np.sum(self.periodicity)
        n_atom = len(self.atoms)
        bond_mat = np.zeros((max_image, n_atom, n_atom))
        dist_mat = self.get_dist_matrix()
        bond_mat = dist_mat < self.bond_length
        bond_mat_2 = dist_mat > 0

        return bond_mat * bond_mat_2
    
    def get_dist_matrix(self):
        dist_mat_vec = self.get_dist_matrix_vec()
        dist_mat = np.linalg.norm(dist_mat_vec, axis=-1)
        return dist_mat

    def get_dist_matrix_vec(self):
        def get_dist_vec(pos1, pos2, lat_vecs, l_min=False):
            """ # p1, p2 direct 
                # return angstrom
                # latConst is included in lat_vecs
            """
            diff = np.array(pos1) - np.array(pos2)
            if np.linalg.norm(diff) ==  0:
                return 0
            if l_min:
                diff = diff - np.round(diff)
            diff = np.dot(lat_vecs.T,diff)
            return diff

        n_atom = len(self.atoms)
        max_image = 3 ** np.sum(self.periodicity)

        lat_vecs = self.lattice.get_matrix()
        atoms = self.atoms
        d_mat = np.zeros((max_image, n_atom, n_atom, 3))

        periodic_image = []
        for period in self.periodicity:
            if period:
                periodic_image.append(np.arange(3) - 1)
            else:
                periodic_image.append([0])

        for image_i, image in enumerate(itertools.product(*periodic_image)):
            for i, atom1 in enumerate(atoms):
                for j, atom2 in enumerate(atoms):
                    diff = get_dist_vec(atom1.pos + np.array(image), atom2.pos, lat_vecs)
                    d_mat[image_i, i, j, :] = diff
        return d_mat


class Lattice:
    """represent lattice of structure
    """
    def __init__(self, *args):
        """
        Args:
            a, b, c, alpha, beta, gamma
        """
        matrix, lat_const = args
        self.a, self.b, self.c, self.alpha, self.beta , self.gamma = \
        self._to_list(matrix, lat_const)

    def _to_list(self, matrix, lat_const):
        """ see http://en.wikipedia.org/wiki/Fractional_coordinates
        """
        from numpy.linalg import norm

        a = matrix[0] * lat_const
        b = matrix[1] * lat_const
        c = matrix[2] * lat_const

        alpha = np.arctan2(norm(np.cross(b, c)), np.dot(b, c))
        beta  = np.arctan2(norm(np.cross(c, a)), np.dot(c, a))
        gamma = np.arctan2(norm(np.cross(a, b)), np.dot(a, b))

        return norm(a), norm(b), norm(c), alpha, beta, gamma

    def get_matrix(self):
        matrix = self._to_matrix()
        return matrix

    def _to_matrix(self):
        # see http://en.wikipedia.org/wiki/Fractional_coordinates
        # For the special case of a monoclinic cell (a common case) where alpha = gamma = 90 degree and beta > 90 degree, this gives: <- special care needed
        # so far, alpha, beta, gamma < 90 degree
        a, b, c, alpha, beta, gamma = self.a, self.b, self.c, self.alpha, self.beta, self.gamma

        v = a * b * c * np.sqrt(1. - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) )

        T = np.zeros((3, 3))
        T = np.array([ \
                  [a, b * np.cos(gamma), c * np.cos(beta)                                                  ] ,\
                  [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)] ,\
                  [0, 0                , v / (a * b * np.sin(gamma))                                       ] 
                      ])
        matrix = np.zeros((3, 3))
        matrix[:,0] = np.dot(T, np.array((1, 0, 0)))
        matrix[:,1] = np.dot(T, np.array((0, 1, 0)))
        matrix[:,2] = np.dot(T, np.array((0, 0, 1)))
        return matrix.T

    def __repr__(self):
        _repr = [self.a, self.b, self.c, self.alpha, self.beta , self.gamma]
        _repr = [str(i) for i in _repr]
        return ' '.join(_repr)


class Atom:
    def __init__(self, element, pos, dyn=None):
        """
        Object to represent atom
        Args:
            element:
                atomic symbol eg) 'Si'
            pos:
                atom position (fractional coordinate) eg) [0.5, 0.5, 0] 
        """
        self.element = element
        self.pos = np.array(pos)

    def to_list(self):
        out_list = [self.element, self.pos, self.dyn]

        return out_list


def main():
    
    a = 5.47
    lat = Lattice(np.array([1 / 2. * np.array([1., 1., 0.]),
                            1 / 2. * np.array([0., 1., 1.]),
                            1 / 2. * np.array([1., 0., 1.])]), a)
    atoms = [Atom('Si', np.array([0., 0., 0.])), Atom('Si', np.array([1., 1., 1.])/4.)]
    fcc_si = Structure(lat, atoms, name='fcc_si')

    b_mat = fcc_si.get_bond_mat()
    print np.sum(b_mat) # should be 4 + 4 = 8

if __name__ == '__main__':
    main()
