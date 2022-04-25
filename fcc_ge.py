#/usr/local/python
import sys_input
import numpy as np
from hamiltonian import Hamiltonian
from band_plotter import draw_band

def main():
    # fcc_ge_input = sys_input.Sys_input('./input_Jancu_PRB_76_115202_2007.yaml')
    fcc_ge_input = sys_input.Sys_input('./input_PRB_57_1998_Jancu.yaml')
    fcc_ge_sys = fcc_ge_input.system
    k_all_path, kpts_len = fcc_ge_input.get_kpts()

    fcc_ge_ham = Hamiltonian(fcc_ge_sys)
    
    eigs_k = []

    for kpt in k_all_path:
        ham = fcc_ge_ham.get_ham(np.array(kpt))
        eigs = np.linalg.eigvalsh(ham)
        eigs_k.append(eigs)

    eigs_k = np.array(eigs_k).T
    print(eigs_k[:,-1])
    draw_band(kpts_len, eigs_k)


if __name__ == '__main__':
    main()