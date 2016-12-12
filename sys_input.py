import yaml
import system
from band_plotter import get_kpt_path, get_kpt_len

class Sys_input():
    def __init__(self, input_file_name='./input.yaml'):
        with open(input_file_name) as _:
            conf = yaml.safe_load(_)

            self.name = conf['struct']['name']
            # read structure
            self.struct_path = conf['struct']['POSCAR_path']
            self.periodicity = conf['struct']['periodicity']
            self.bond_cut = conf['struct']['bond_cut']

            # read kpoints path
            self.kpt_den = conf['kpt_paths']['kpt_den']
            self.sp_kpts = conf['kpt_paths']['sp_kpts']

            self.orbitals = conf['orbitals']

            self.params = conf['params']
            self.scale_params = conf['scale_params']

            self.system = self._get_system()

    def _get_system(self):
        kwargs = {'periodicity': self.periodicity,
                  'name': self.name,
                  'bond_cut': self.bond_cut}
        struct = system.Structure.read_poscar(self.struct_path, kwargs)

        sys = system.System(struct, self.orbitals, self.params, self.scale_params)
        return sys

    def get_kpts(self):

        sp_kpts = self.sp_kpts
        kpt_den = self.kpt_den
        
        kpt_path = get_kpt_path(sp_kpts, kpt_den)
        kpts_len = get_kpt_len(kpt_path, self.system.structure.lattice.get_matrix())
        k_all_path = [kpt for kpt_path_seg in kpt_path
                          for kpt in kpt_path_seg]
        return k_all_path, kpts_len


if __name__ == '__main__':
    """ test """
    sys_input = Sys_input('./input_Jancu_PRB_76_115202_2007.yaml')
    sys = sys_input.system

    print sys.get_soc_mat()
    # struct = sys.structure
