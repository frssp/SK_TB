import numpy as np
import itertools


class System(object):
    """ atomics structures and tight_binding parameters 
    """
    def __init__(self, structure, orbitals, parameters):
        self.structure = structure
        self.orbitals = orbitals
        self.set_orbitals()
        self.all_orbitals = self.get_all_orbitals()
        self.all_iter = self.get_all_iter()
        self.params = parameters
        assert self.params.keys() == self.get_param_key(), 'wrong parameter set'

    def set_orbitals(self):
        for atom in self.structure.atoms:
            atom.set_orbitals(self.orbitals[atom.element])

    def get_all_orbitals(self):
        all_orbitals = []
        for atom in self.structure.atoms:
            for orbit in atom.orbitals:
                all_orbitals.append((atom.element, orbit))
        return all_orbitals

    def get_all_iter(self):
        all_orbitals = []
        for atom_i, atom in enumerate(self.structure.atoms):
            for orbit_i, orbit in enumerate(atom.orbitals):
                all_orbitals.append((atom_i, orbit_i, atom.element, orbit))
        return all_orbitals

    def get_param_key(self):
        elements = self.structure.get_elements()
        key_list = []
        key_list += elements
        for key in itertools.product(elements, repeat=2):
            key_list.append(''.join(key))
        return key_list


class Structure(object):
    """ atomic structure
    """

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
        self.max_image = 3 ** np.sum(self.periodicity)

        self.bond_mat = self.get_bond_mat()
        self.dist_mat_vec = self.get_dist_matrix_vec()

    def get_bond_mat(self):
        max_image = self.max_image
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
        max_image = self.max_image

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

    def get_elements(self):
        """return list of elements eg) ['Si', 'O']"""
        from collections import OrderedDict
        return list(OrderedDict.fromkeys([atom.element for atom in self.atoms]))


class Lattice:
    """represent lattice of structure
    """
    def __init__(self, *args):
        """
        Args:
            a, b, c, alpha, beta, gamma
        """
        matrix, lat_const = args
        self.matrix = np.array(matrix) * lat_const
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
        # return matrix.T
        return self.matrix

    def get_rec_lattice(self):
        """
        b_i = (a_j x a_k)/ a_i . (a_j x a_k)
        """
        lat_mat = self.matrix
        rec_lat_mat = np.linalg.inv(lat_mat).T
        return rec_lat_mat

    def __repr__(self):
        _repr = [self.a, self.b, self.c, self.alpha, self.beta , self.gamma]
        _repr = [str(i) for i in _repr]
        return ' '.join(_repr)


class Atom:
    ORBITALS_ALL = ['s',
                    'px', 'py', 'pz',
                    'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
                    'S']
    def __init__(self, element, pos):
        """ Object to represent atom
            Args:
                element:
                    atomic symbol eg) 'Si'
                pos:
                    atom position (fractional coordinate) eg) [0.5, 0.5, 0] 
                orbitals:
                    subset of ['s',
                               'px', 'py', 'pz',
                               'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2',
                               'S']
        """

        self.element = element
        self.pos = np.array(pos)
        self.orbitals = None

    def to_list(self):
        out_list = [self.element, self.pos, self.dyn]

        return out_list

    def set_orbitals(self, orbitals=None):
        assert set(orbitals).issubset(set(Atom.ORBITALS_ALL)), 'wrong orbitals'
        self.orbitals = orbitals


def fcc_ge_sys():
    # params = {
    #     'Ge': {
    #         'es': 0,
    #         'ep': 7.20,
    #     },
    #     'GeGe': {
    #         'Vsss': -8.13 / 4.,
    #         'Vsps': 5.88 * np.sqrt(3) / 4.,
    #         'Vpps': (3.17 + 2 * 7.51) / 4. ,
    #         'Vppp': (3.17 - 7.51) / 4.,
    #     }
    # }
    # params = {
    #     'Ge': {
    #         'es': 0,
    #         'ep': 8.41,
    #     },
    #     'GeGe': {
    #         'Vsss': -6.78 / 4.,
    #         'Vsps': 5.91 * np.sqrt(3) / 4.,
    #         'Vpps': (2.62 + 2 * 6.82) / 4. ,
    #         'Vppp': (2.62 - 6.82) / 4.,
    #     }
    # }
    params = {
        'Ge': {
            'es': -2.55247,
            'ep': 4.48593,
            'ed': 14.01053,
            'eS': 23.44607,
        },
        'GeGe': {
            'Vsss': -1.86600,
            'Vsps': 2.91067,
            'Vppp': -1.49207,
            'Vpps': 4.08481,

            'Vsds': -2.23992,
            'Vpds': -1.66657,
            'Vpdp': 2.39936,
            'Vdds': -1.82945,
            'Vddp': 3.08177,
            'Vddd': -1.56676,
            
            'VSSs': -4.51331,
            'VsSs': -1.39107,
            'VSps': 3.06822,
            'VSds': -0.77711
        }
    }
    params = {
        'Ge': {
            'es': -3.2967,
            'ep': 4.6560,
            'ed': 13.0143,
            'eS': 19.1725,
        },
        'GeGe': {
            'Vsss': -1.5003,
            'VSSs': -3.6029,
            'VsSs': -1.9206,

            'Vsps': 2.7986,
            'VSps': 2.8177,
            'Vsds': -2.8028,
            'VSds': -0.6209,
            
            'Vpps': 4.2541,
            'Vppp': -1.6510,

            'Vpds': -2.2138,
            'Vpdp': 1.9001,

            'Vdds': -1.2172,
            'Vddp': 2.5054,
            'Vddd': -2.1389,
            
        }
    }
    orbitals = {'Ge': ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dxz', 'dx2-y2', 'dz2', 'S']}
    # orbitals = {'Ge': ['s', 'px', 'py', 'pz', 'S']}
    # orbitals = {'Si': ['s']}
    fcc_ge = fcc_ge_struct()
    fcc_ge_sys = System(fcc_ge, orbitals, params)
    return fcc_ge_sys

def fcc_ge_struct():
    a = 5.43
    lat = Lattice(np.array([1 / 2. * np.array([1., 1., 0.]),
                            1 / 2. * np.array([0., 1., 1.]),
                            1 / 2. * np.array([1., 0., 1.])]), a)
    atoms = [Atom('Ge', np.array([0., 0., 0.])), 
             Atom('Ge', np.array([1., 1., 1.])/4.)]
    fcc_ge = Structure(lat, atoms, name='fcc_ge')
    return fcc_ge


if __name__ == '__main__':
    fcc_ge_sys()
