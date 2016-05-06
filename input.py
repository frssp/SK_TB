import yaml
import system

class sys_input():
    def __init__(self, input_file_name='./input.yaml'):
        with open(input_file_name) as _:
            conf = yaml.safe_load(_)

            self.name = conf['struct']['name']
            # read structure
            self.struct_path = conf['struct']['POSCAR_path']
            self.periodicity = conf['struct']['periodicity']
            self.NN_length = conf['struct']['NN_length']
            self.orbitals = conf['orbitals']

            self.params = conf['params']
            self.scale_params = conf['scale_params']

    def get_system(self):
        kwargs = {'NN_length': self.NN_length,
                  'periodicity': self.periodicity,
                  'name': self.name}
        struct = system.Structure.read_poscar(self.struct_path, kwargs)

        sys = system.System(struct, self.orbitals, self.params, self.scale_params)
        return sys


if __name__ == '__main__':
    """ test """
    sys_input = sys_input('./input.yaml')
    sys_input.get_system()
    