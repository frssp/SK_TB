from sympy import symbols
from numpy import sqrt
(l, m, n) = symbols('l m n')
(Vsss, Vsps, Vpps, Vppp) = symbols('Vsss Vsps Vpps Vppp')
(Vsds, Vpds, Vpdp, Vdds, Vddp, Vddd) = symbols('Vsds Vpds Vpdp Vdds Vddp Vddd')
(VSSs, VsSs, VSps, VSds) = symbols('VSSs, VsSs, VSps, VSds')

#[s, px, py, pz, dxy, dyz, dxz, dx2-y2, dz2, S]
HOPPING_INTEGRALS = [[l * 0 for _ in xrange(10)] for __ in xrange(10)]

HOPPING_INTEGRALS[0][0] = Vsss
HOPPING_INTEGRALS[0][1] = l * Vsps
HOPPING_INTEGRALS[0][2] = m * Vsps
HOPPING_INTEGRALS[0][3] = n * Vsps
HOPPING_INTEGRALS[1][0] = -HOPPING_INTEGRALS[0][1]
HOPPING_INTEGRALS[2][0] = -HOPPING_INTEGRALS[0][2]
HOPPING_INTEGRALS[3][0] = -HOPPING_INTEGRALS[0][3]
HOPPING_INTEGRALS[0][4] = sqrt(3) * l * m * Vsds
HOPPING_INTEGRALS[0][5] = sqrt(3) * m * n * Vsds
HOPPING_INTEGRALS[0][6] = sqrt(3) * l * n * Vsds
HOPPING_INTEGRALS[4][0] = HOPPING_INTEGRALS[0][4]
HOPPING_INTEGRALS[5][0] = HOPPING_INTEGRALS[0][5]
HOPPING_INTEGRALS[6][0] = HOPPING_INTEGRALS[0][6]
HOPPING_INTEGRALS[0][7] = sqrt(3) / 2. * (l ** 2 - m ** 2) * Vsds
HOPPING_INTEGRALS[7][0] = HOPPING_INTEGRALS[0][7]
HOPPING_INTEGRALS[0][8] = (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * Vsds
HOPPING_INTEGRALS[8][0] = HOPPING_INTEGRALS[0][8]

HOPPING_INTEGRALS[1][1] = l ** 2 * Vpps + (1. - l ** 2) * Vppp
HOPPING_INTEGRALS[1][2] = l * m * (Vpps - Vppp)
HOPPING_INTEGRALS[2][1] = HOPPING_INTEGRALS[1][2]
HOPPING_INTEGRALS[1][3] = l * n * (Vpps - Vppp)
HOPPING_INTEGRALS[3][1] = HOPPING_INTEGRALS[1][3]

HOPPING_INTEGRALS[1][4] = sqrt(3) * l ** 2 * m * Vpds + m * (1. - 2 * l ** 2) * Vpdp
HOPPING_INTEGRALS[1][5] = l * m * n * (sqrt(3) * Vpds - 2 * Vpdp)
HOPPING_INTEGRALS[1][6] = sqrt(3) * l ** 2 * n * Vpds + n * (1. - 2 * l ** 2) * Vpdp
HOPPING_INTEGRALS[4][1] = -HOPPING_INTEGRALS[1][4]
HOPPING_INTEGRALS[5][1] = -HOPPING_INTEGRALS[1][5]
HOPPING_INTEGRALS[6][1] = -HOPPING_INTEGRALS[1][6]

HOPPING_INTEGRALS[1][7] = 0.5 * sqrt(3) * l * (l ** 2 - m ** 2) * Vpds + l * (1. - l ** 2 + m ** 2) * Vpdp
HOPPING_INTEGRALS[1][8] = l * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * Vpds - sqrt(3) * l * n ** 2 * Vpdp
HOPPING_INTEGRALS[7][1] = -HOPPING_INTEGRALS[1][7]
HOPPING_INTEGRALS[8][1] = -HOPPING_INTEGRALS[1][8]

HOPPING_INTEGRALS[2][2] = m ** 2 * Vpps + (1. - m ** 2) * Vppp
HOPPING_INTEGRALS[2][3] = m * n * (Vpps - Vppp)
HOPPING_INTEGRALS[3][2] = HOPPING_INTEGRALS[2][3]

HOPPING_INTEGRALS[2][4] = sqrt(3) * m ** 2 * l * Vpds + l * (1. - 2 * m ** 2) * Vpdp
HOPPING_INTEGRALS[2][5] = sqrt(3) * m ** 2 * n * Vpds + n * (1. - 2 * m ** 2) * Vpdp
HOPPING_INTEGRALS[2][6] = l * m * n * (sqrt(3) * Vpds - 2 * Vpdp)
HOPPING_INTEGRALS[4][2] = -HOPPING_INTEGRALS[2][4]
HOPPING_INTEGRALS[5][2] = -HOPPING_INTEGRALS[2][5]
HOPPING_INTEGRALS[6][2] = -HOPPING_INTEGRALS[2][6]

HOPPING_INTEGRALS[2][7] = 0.5 * sqrt(3) * m * (l ** 2 - m ** 2) * Vpds - m * (1. + l ** 2 - m ** 2) * Vpdp
HOPPING_INTEGRALS[2][8] = m * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * Vpds - sqrt(3) * m * n ** 2 * Vpdp
HOPPING_INTEGRALS[7][2] = -HOPPING_INTEGRALS[2][7]
HOPPING_INTEGRALS[8][2] = -HOPPING_INTEGRALS[2][8]

HOPPING_INTEGRALS[3][3] = n ** 2 * Vpps + (1. - n ** 2) * Vppp

HOPPING_INTEGRALS[3][4] = l * m * n * (sqrt(3) * Vpds - 2 * Vpdp)
HOPPING_INTEGRALS[3][5] = sqrt(3) * n ** 2 * m * Vpds + m * (1. - 2 * n ** 2) * Vpdp
HOPPING_INTEGRALS[3][6] = sqrt(3) * n ** 2 * l * Vpds + l * (1. - 2 * n ** 2) * Vpdp
HOPPING_INTEGRALS[4][3] = -HOPPING_INTEGRALS[3][4]
HOPPING_INTEGRALS[5][3] = -HOPPING_INTEGRALS[3][5]
HOPPING_INTEGRALS[6][3] = -HOPPING_INTEGRALS[3][6]

HOPPING_INTEGRALS[3][7] = 0.5 * sqrt(3) * n * (l ** 2 - m ** 2) * Vpds - n * (l ** 2 - m ** 2) * Vpdp
HOPPING_INTEGRALS[3][8] = n * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * Vpds + sqrt(3) * n * (l ** 2 + m ** 2) * Vpdp
HOPPING_INTEGRALS[7][3] = -HOPPING_INTEGRALS[3][7]
HOPPING_INTEGRALS[8][3] = -HOPPING_INTEGRALS[3][8]

HOPPING_INTEGRALS[4][4] = l ** 2 * m ** 2 * (3 * Vdds - 4 * Vddp + Vddd) + (l ** 2 + m ** 2) * Vddp + n ** 2 * Vddd
HOPPING_INTEGRALS[5][5] = m ** 2 * n ** 2 * (3 * Vdds - 4 * Vddp + Vddd) + (m ** 2 + n ** 2) * Vddp + l ** 2 * Vddd
HOPPING_INTEGRALS[6][6] = n ** 2 * l ** 2 * (3 * Vdds - 4 * Vddp + Vddd) + (n ** 2 + l ** 2) * Vddp + m ** 2 * Vddd

HOPPING_INTEGRALS[4][5] = l * m ** 2 * n * (3 * Vdds - 4 * Vddp + Vddd) + l * n * (Vddp - Vddd)
HOPPING_INTEGRALS[4][6] = n * l ** 2 * m * (3 * Vdds - 4 * Vddp + Vddd) + n * m * (Vddp - Vddd)
HOPPING_INTEGRALS[5][6] = m * n ** 2 * l * (3 * Vdds - 4 * Vddp + Vddd) + m * l * (Vddp - Vddd)
HOPPING_INTEGRALS[5][4] = HOPPING_INTEGRALS[4][5]
HOPPING_INTEGRALS[6][4] = HOPPING_INTEGRALS[4][6]
HOPPING_INTEGRALS[6][5] = HOPPING_INTEGRALS[5][6]

HOPPING_INTEGRALS[4][7] = 0.5 * l * m * (l ** 2 - m ** 2) * (3 * Vdds - 4 * Vddp + Vddd)
HOPPING_INTEGRALS[5][7] = 0.5 * m * n *((l ** 2 - m ** 2) * (3 * Vdds - 4 * Vddp + Vddd) - 2 * (Vddp - Vddd))
HOPPING_INTEGRALS[6][7] = 0.5 * n * l *((l ** 2 - m ** 2) * (3 * Vdds - 4 * Vddp + Vddd) + 2 * (Vddp - Vddd))

HOPPING_INTEGRALS[7][4] = HOPPING_INTEGRALS[4][7]
HOPPING_INTEGRALS[7][5] = HOPPING_INTEGRALS[5][7]
HOPPING_INTEGRALS[7][6] = HOPPING_INTEGRALS[6][7]

HOPPING_INTEGRALS[4][8] = sqrt(3) * (l * m * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * Vdds - 2 * l * m * n ** 2 * Vddp + 0.5 * l * m * (1. + n ** 2) * Vddd)
HOPPING_INTEGRALS[5][8] = sqrt(3) * (m * n * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * Vdds + m * n * (l ** 2 + m ** 2 - n ** 2) * Vddp - 0.5 * m * n * (l ** 2 + m ** 2) * Vddd)
HOPPING_INTEGRALS[6][8] = sqrt(3) * (n * l * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * Vdds + n * l * (l ** 2 + m ** 2 - n ** 2) * Vddp - 0.5 * n * l * (l ** 2 + m ** 2) * Vddd)
HOPPING_INTEGRALS[8][4] = HOPPING_INTEGRALS[4][8]
HOPPING_INTEGRALS[8][5] = HOPPING_INTEGRALS[5][8]
HOPPING_INTEGRALS[8][6] = HOPPING_INTEGRALS[6][8]

HOPPING_INTEGRALS[7][7] = 0.25 * (l ** 2 - m ** 2) ** 2 * (3 * Vdds - 4 * Vddp + Vddd) + (l ** 2 + m ** 2) * Vddp + n ** 2 * Vddd
HOPPING_INTEGRALS[8][8] = 0.75 * (l ** 2 + m ** 2) ** 2 * Vddd + 3 * (l ** 2 + m ** 2) * n ** 2 * Vddp + 0.25 * (l ** 2 + m ** 2 - 2*n ** 2) ** 2 * Vdds
HOPPING_INTEGRALS[7][8] = 0.25 * (l ** 2 - m ** 2) * (n ** 2 * (3 * Vdds - 4 * Vddp + Vddd) + Vddd - Vdds)
HOPPING_INTEGRALS[8][7] = 0.25 * (l ** 2 - m ** 2) * (n ** 2 * (3 * Vdds - 4 * Vddp + Vddd) + Vddd - Vdds)

HOPPING_INTEGRALS[9][9] = VSSs
HOPPING_INTEGRALS[0][9] = VsSs
HOPPING_INTEGRALS[9][0] = VsSs
HOPPING_INTEGRALS[9][1] = l * VSps
HOPPING_INTEGRALS[9][2] = m * VSps
HOPPING_INTEGRALS[9][3] = n * VSps
HOPPING_INTEGRALS[1][9] = -HOPPING_INTEGRALS[9][1]
HOPPING_INTEGRALS[2][9] = -HOPPING_INTEGRALS[9][2]
HOPPING_INTEGRALS[3][9] = -HOPPING_INTEGRALS[9][3]
HOPPING_INTEGRALS[9][4] = sqrt(3) * l * m * VSds
HOPPING_INTEGRALS[9][5] = sqrt(3) * m * n * VSds
HOPPING_INTEGRALS[9][6] = sqrt(3) * l * n * VSds
HOPPING_INTEGRALS[4][9] = HOPPING_INTEGRALS[9][4]
HOPPING_INTEGRALS[5][9] = HOPPING_INTEGRALS[9][5]
HOPPING_INTEGRALS[6][9] = HOPPING_INTEGRALS[9][6]
HOPPING_INTEGRALS[9][7] = sqrt(3) / 2. * (l * l - m * m) * VSds
HOPPING_INTEGRALS[7][9] = HOPPING_INTEGRALS[9][7]
HOPPING_INTEGRALS[9][8] = (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * VSds
HOPPING_INTEGRALS[8][9] = HOPPING_INTEGRALS[9][8]


