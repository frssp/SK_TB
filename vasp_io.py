'''
Created on 2014.  05.  14.
 
@author: nwan
@author: hanwooh
'''
import numpy as np 
import re

FLOAT_REG = '[-+]?\d*\.?\d+'

def findReg(reg, file):
    f = open(file)
    lines = f.readlines()
    f.close()
    find = []
    for line in lines:
        m = re.findall(reg, line)
        find += m
    return find


def readPRJCAR(file_name):
    def _parse_kpt(line):
        '''
        k-point (associated with POSCAR):     2  vkpt:     
        0.2000000     0.0000000     0.0000000  
        weight:     0.0640000
        '''
        reg = '(\d+)\s*vkpt:\s+({0})\s+({0})\s+({0})\s+weight:\s+({0})'.format(FLOAT_REG)
        match = re.search(reg, line)
        if match is not None:
            temp = [float(item) for item in match.groups()]
            temp[0] = int(temp[0])
            return temp[1:]
        else:
            return None

    try:
        with open(file_name) as prjcar:
            head = prjcar.readline()
            lat_mat_prim = []
            for _ in range(3):
                vec = [float(item) for item in prjcar.readline().split()]
                lat_mat_prim.append(vec)
            lat_mat_prim = np.array(lat_mat_prim)
            prjcar.readline()
            n_kpts_prim = int(prjcar.readline().split()[-1])
            prjcar.readline()
            prjcar.readline()

            # read kpoints
            kpt_list_prim = []
            k_wt_list_prim = []

            for i in range(n_kpts_prim):
                line = prjcar.readline().split()
                wt = int(line[-1])
                kpt = np.array(line[1:4], float)
                k_wt_list_prim.append(wt)
                kpt_list_prim.append(kpt)
            prjcar.readline()

            # read projection sum_G <k'G'|psi_nk>
            kpt_list = []
            k_wt_list = []
            energy_list = []
            K_list = []
            # ignore spin so far
            # kpt loop 
            line = prjcar.readline()
            while True:
                # print line
                # break
                if not line: break
                if 'k-point' in line:
                    # print line
                    kpt = _parse_kpt(line)
                    wt = kpt[-1]
                    kpt = kpt[:-1]
                    kpt_list.append(kpt)
                    k_wt_list.append(wt)

                    # read energy
                    # print line
                    while True:
                        line = prjcar.readline()
                        # print line
                        if not 'band' in line:
                            break
                        if not line: break
                        energy = float(line.split()[-1])
                        energy_list.append(energy)

                        # read K
                        K = []
                        for i in range(int(np.ceil(n_kpts_prim / 10.))):
                            line = prjcar.readline().split()
                            K += line
                        K = np.ravel(np.array(K, float))
                        K_list.append(K)
                else:
                    line = prjcar.readline()
            # print len(energy_list)
            # print len(K_list)
            # print len(kpt_list), len(energy_list) / len(kpt_list)
            shape_e = (len(kpt_list), len(energy_list) / len(kpt_list))
            shape_K = (len(kpt_list), len(energy_list) / len(kpt_list), n_kpts_prim)

            energy_list = np.array(energy_list).reshape(shape_e)
            K_list = np.array(K_list).reshape(shape_K)
            return kpt_list_prim, k_wt_list_prim, energy_list, K_list, kpt_list
    except  e:
        print(e)
        raise e


def readPROCAR_phase(file_name):
    """
    read PROCAR file of VASP 
    """
    # nested parsing functions
    def _parse_procar_metadata(procar):
        '''
        from file object procar
        read n_kpts, n_bands, n_ion, and n_orbit, orbit_names
        seek back file pointer to original position
        '''
        f_pointer = procar.tell()

        # read n_kpts, n_bands, n_ion
        ' # of k-points:   14         # of bands: 450         # of ions:  85 '
        for line in procar:
            reg = ('# of k-points:\s*(\d+)\s*'
                   '# of bands:\s*(\d+)\s*'
                   '# of ions:\s*(\d+)')
            match = re.search(reg, line)
            if match is not None:
                n_kpts, n_bands, n_ions = [int(item) for item in match.groups()]
                break

        # read n_orbit
        'ion      s     py     pz     px    dxy    dyz    dz2    dxz    dx2    tot'
        for line in procar:
            reg = '^ion.*tot$'
            match = re.search(reg, line)
            if match is not None:
                orbit_names = match.group().split()[1:-1]
                # print orbit_names
                n_orbits = len(orbit_names)
                break

        procar.seek(f_pointer)
        return n_kpts, n_bands, n_ions, n_orbits, orbit_names

    def _parse_kpt(line):
        ' k-point    1 :    0.00000000 0.00000000 0.00000000     weight = 0.25000000'
        reg = ' k-point\s*(\d*) :\s*({0}) ({0}) ({0})\s* weight = ({0})'.format(FLOAT_REG)
        match = re.search(reg, line)
        if match is not None:
            temp = [float(item) for item in match.groups()]
            temp[0] = int(temp[0])
            return temp
        else:
            return None

    def _parse_eigenval(line):
        'band   1 # energy  -13.72545072 # occ.  2.00000000'
        reg = 'band\s*(\d*)\s*# energy\s*({0})\s*# occ.\s*({0})'.format(FLOAT_REG)
        match = re.search(reg, line)
        if match is not None:
            temp = [float(item) for item in match.groups()]
            temp[0] = int(temp[0])
            return temp
        else:
            return None

    def _parse_proj_abs(line, n_orbits):
        '  1  0.005  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.005'
        reg = '^(\s*\d+)\s+({})\s+{}'.format((FLOAT_REG + '\s+') * n_orbits, FLOAT_REG)
        match = re.search(reg, line)
        if match is not None:
            ion_i, proj = match.groups()
            return int(ion_i) -1 , [float(item) for item in proj.split()]
        else:
            return None

    def _parse_proj_phase(line, n_orbits):
        """
          1 -0.123  0.029 -0.002 -0.000  0.000  0.000  0.000  0.000  0.000
          1 -0.093  0.022 -0.002  0.000  0.000  0.000  0.000  0.000  0.000
        """
        reg = '^(\s*\d+)\s+({})'.format((FLOAT_REG + '\s+') * n_orbits)
        match = re.search(reg, line)
        if match is not None:
            ion_i, proj = match.groups()
            return int(ion_i) - 1, [float(item) for item in proj.split()]
        else:
            return None

    try:
        with open(file_name) as procar:
            # procar = procar.readlines()
            # head = procar[0]
            head = procar.readline()
            l_phase = 'phase' in head

            n_kpts, n_bands, n_ions, n_orbits, orbit_names = \
                _parse_procar_metadata(procar)

            print(n_orbits)

            Proj = np.zeros((n_kpts, n_bands, n_ions, n_orbits), float)
            Proj_cmplx = np.zeros((n_kpts, n_bands, n_ions, n_orbits), complex)
            Kpts = np.zeros((n_kpts, 3))
            k_wt = np.zeros(n_kpts)
            Eigs = np.zeros((n_kpts, n_bands))
            Occs = np.zeros((n_kpts, n_bands))

            kpt_i = 0
            band_i = 0
            ion_i = 0

            # for line in procar:
            while True:
                line = procar.readline()
                if not line: break

                # read kpt, wight
                kpt = _parse_kpt(line)
                if kpt is not None:
                    kpt_i, kpt, weight = kpt[0] - 1, kpt[1:4], kpt[4]
                    # print kpt_i, kpt, weight
                    Kpts[kpt_i, :] = kpt
                    k_wt[kpt_i] = weight
                    continue

                # read eig, occ
                eig = _parse_eigenval(line)
                if eig is not None:
                    band_i, eig, occ = eig[0] - 1, eig[1], eig[2]
                    Eigs[kpt_i, band_i] = eig
                    Occs[kpt_i, band_i] = occ
                    continue

                # read absolute projection
                proj = _parse_proj_abs(line, n_orbits)
                if proj is not None:
                    ion_i, proj = proj
                    Proj[kpt_i, band_i, ion_i, :] = proj
                    continue

                # read projection with phase
                proj_cmplx = _parse_proj_phase(line, n_orbits)
                # print procar.readline()
                # print procar.readline()
                if proj_cmplx is not None:
                    ion_i, proj_re = proj_cmplx
                    ion_i, proj_im = _parse_proj_phase(procar.readline(), n_orbits)
                    Proj_cmplx[kpt_i, band_i, ion_i, :] = \
                        proj_re + np.array(proj_im) * 1j
                    continue

        if l_phase:
            return Kpts, k_wt, Eigs, Proj, Proj_cmplx, Occs
        else:
            return Kpts, k_wt, Eigs, Proj, Occs
    except e:
        print (e)
        print (line)
        print ('fail to read {}'.format(file_name))


def readPROCAR(fileName='PROCAR', orbital=-1):
    f = open(fileName)
    buffer = f.readlines()

    ''' # of k-points:   14         # of bands: 450         # of ions:  85 '''
    nKpt   = int(re.search('(?<=# of k-points:)\s+\d+',buffer[1]).group(0) )
    nBands = int(re.search('(?<=# of bands:)\s+\d+'   ,buffer[1]).group(0) )
    nIons  = int(re.search('(?<=# of ions:)\s+\d+'    ,buffer[1]).group(0) )
    
    nOrbits = len(buffer[7].split())-1
    Proj = np.zeros((nKpt,nBands,nIons,nOrbits))
    Kpts = np.zeros((nKpt,3))
    Eigs = np.zeros((nKpt,nBands))
    Occs = np.zeros((nKpt,nBands))

    kptInfoLength =1
    for i in [line.find(' k-point') for line in buffer[3+1:]]:
        kptInfoLength-=i
        if i==0:break


    for kpt in range(nKpt):
        # read k-th band projection to ion orbital 
        # read k-point
        kptLineNum = 2 + 1 + kptInfoLength* kpt
        kptLine = buffer[kptLineNum]

        kVec = re.search('(?<=:)\s*([-]?[0-9]*\.?[0-9]+)\s*([-]?[0-9]*\.?[0-9]+)\s*([-]?[0-9]*\.?[0-9]+)',kptLine)
        kVec = np.array( [float(kVec.group(1)), float(kVec.group(2)), float(kVec.group(3))] )
        Kpts[kpt,:] = kVec
        
        kp_weight = float( re.search('(?<=weight =)\s*[-]?[0-9]*\.?[0-9]+',kptLine).group(0) )

        bandInfoLength =1
        for i in [line.find('band') for line in buffer[kptLineNum+2+1:]]:
            bandInfoLength-=i
            if i==0:break
        # print bandInfoLength

        for band in range(nBands): 
            # bandLineNum = kptLineNum + 2 + band * (nIons *3 +6) #works for only non-S*L coupling
            bandLineNum = kptLineNum + 2 + band *bandInfoLength
            eig = float( re.search('(?<=energy)\s+[-]?[0-9]*\.?[0-9]+',buffer[bandLineNum]).group(0) )
            occ = float( re.search('(?<=occ.)\s+[-]?[0-9]*\.?[0-9]+',buffer[bandLineNum]).group(0) )
            
            Eigs[kpt,band] = eig
            Occs[kpt,band] = occ

            for ion in range(nIons):
                ionLineNum = bandLineNum +3 + ion

                orbital_proj = [float(o) for o in buffer[ionLineNum].split()[1:]]
                Proj[kpt,band,ion,:] = orbital_proj

    return Kpts, Eigs, Proj, Occs


def readPOSCAR(fileName='POSCAR', rtspecies=False):
    return readCONTCAR(fileName, rtspecies)


def readCONTCAR(fileName='CONTCAR', rtspecies=False, rt_comment=False):
    latticeVecs=[]
    atomSet=[]
    atomSetDirect=[]
    dynamics_list = []
    sd=False
    f=open(fileName,'r')
    # read first & second line 
    comment = f.readline()
    latConst = float(f.readline())
    # read lattice vectors
    latVec = np.array([float(i)*latConst for i in f.readline().split()])
    latticeVecs.append(latVec)
    latVec = np.array([float(i)*latConst for i in f.readline().split()])
    latticeVecs.append(latVec)
    latVec = np.array([float(i)*latConst for i in f.readline().split()])
    latticeVecs.append(latVec)
    
    # read species
    species=f.readline().split()
    numSpecies=[int(i) for i in f.readline().split()]
    
    line = f.readline().strip()

    if line == 'Selective dynamics':
        l_selective = True      
        DorC = f.readline()
    else:
        l_selective = False
        DorC = line
    
    # read coordinate 
    k=0
    for symbol in species:
        for n in range(numSpecies[k]):
            line = f.readline()
            coord = np.array([float(i) for i in line.split()[:3]])
            if l_selective:
                dynamics = [l_dyn for l_dyn in line.split()[3:]]
            else:
                dynamics = [True, True, True]
            atomSetDirect.append([symbol,coord])
            dynamics_list.append(dynamics)

            if DorC[0]=='D' or  DorC[0]=='d' : # Direct
                coord = latticeVecs[0]*coord[0]+latticeVecs[1]*coord[1]+latticeVecs[2]*coord[2]
            else: 
                print ("check coord! it's not direct form")

            atomSet.append([symbol,coord])
        k += 1
    f.close()

    for i,latVec in enumerate(latticeVecs):
        latticeVecs[i]= latVec / latConst

    return_list = [latConst, latticeVecs, atomSetDirect, dynamics_list]
    
    if rtspecies==True:
        return_list.append(species)
    if rt_comment==True:
        return_list.append(comment)

    return return_list


def writeKPOINTS(fileName='KPOINTS', n_kpt=None):
    """
    write kpoints in MP format
    """
    n_kpt = n_kpt or [1, 1, 1]
    with open(fileName, 'w') as kpoints_file:
        kpoints_file.write('vasp_io generated kpoints mesh\n')
        kpoints_file.write('0\n')
        kpoints_file.write('Gamma\n')
        kpoints_file.write('{} {} {}\n'.format(*n_kpt))


def readLOCPOT(fileName='LOCPOT'):
    return readCHGCAR(fileName)


def readCHGCAR(file_name='CHGCAR'):
    '''
    read CHGCAR
    '''
    import math
    # import itertools
    VALUE_PER_LINE = 5
    with open(file_name) as chgcar_file:
        buffer = chgcar_file.readlines()
        lat_const = float(buffer[1])
        lattice_matrix = np.array(
                         [[float(item) for item in line.split()] 
                           for line in buffer[2:5]])
        n_atom = sum([int(i) for i in buffer[6].split()])
        grids = [int(grid) for grid in buffer[9 + n_atom].split()]

        value = [line for line in buffer[10 + n_atom:
                 10 + n_atom + int(math.ceil(np.prod(grids) / VALUE_PER_LINE))]]
        # avoid read augmentation occupancy
        # value = list(itertools.takewhile(lambda line: 'augmentation' not in line, value))
        value = np.array([float(item) for line in value
                          for item in line.split()])
        CHGCAR = value.reshape(grids[::-1]).T

    return  lat_const, lattice_matrix, CHGCAR


def readEIGENVAL(fileName='EIGENVAL'):
    # read EIGENVAL
    f=open(fileName)
    buffer=f.readlines()
    [nKpt,nBand]=[int(i) for i in buffer[5].split()][1:]
    # print [nBand,nKpt]
    bandInfo = []
    kpoints =[]
    eigenvals =np.zeros((nKpt,nBand))
    #print 'NOW READING ENERGY PART OF EIGENVAL FILE'
    for j in range(nKpt):
        kpoint =np.array(buffer[-1 + 8 + (nBand+2)*j].split())[:3]
        kpoint = np.array([float(k) for k in kpoint])
        kpoints.append(kpoint)

        for i in range(nBand):
            eigenval = buffer[i + 8 + (nBand+2)*j].split()
            eigenval = float(eigenval[1])
            eigenvals[j,i] = eigenval
            #bandInfo.append([i+1,j+1,eigenval])    
    f.close()
    
    return kpoints,eigenvals


def readDOSCAR(fileName, atomNum=None):
    f=open(fileName)
    data=f.readlines()
    f.close()
    numAtom=data[0]
    numAtom=numAtom.split()
    numAtom=int(numAtom[0])
    #print('numAtom',numAtom)
    '''
    read PDOS
    '''
    '''
    go to data of atom selected
    '''
    eSet=[]
    sDOSSet=[]
    pDOSSet=[]
    dDOSSet=[]

    head=data[5].split()
    numRow=int(head[2])
    fermiE=float(head[3])

    eSet=[]
    tDOSSet=[]
    iDOSSet=[]
    '''read total dos'''
    for i in range(6,6+numRow):
        row=data[i].split()
        e=float(row[0])-fermiE
        tDOS=float(row[1])
        iDOS=float(row[2])

        eSet.append(e)
        tDOSSet.append(tDOS)
        iDOSSet.append(iDOS)
    tDOSSet = np.array(tDOSSet)


    ''' read average pdos or specific atom pdos '''
    if atomNum>0:
        atomSet = [atomNum]
    else:
        atomSet = range(numAtom)


    ''' read pdos  '''
    sDOSSetSum = np.zeros(numRow)
    pDOSSetSum = np.zeros(numRow)
    dDOSSetSum = np.zeros(numRow)
    for atomNum_i in range(numAtom):
        # eSet=[]
        sDOSSet=[]
        pDOSSet=[]
        dDOSSet=[]  
        sRow=5+(atomNum_i+1)*(numRow+1)
        head=data[sRow].split()
        
        
        #print('numRow',numRow)

        for i in range(sRow+1,sRow+1+numRow):
            row=data[i].split()
            # e=float(row[0])-fermiE
            sDOS=float(row[1])
            pDOS=float(row[2])
            dDOS=float(row[3])
            # eSet.append(e)
            sDOSSet.append(sDOS)
            pDOSSet.append(pDOS)
            dDOSSet.append(dDOS)
        sDOSSetSum += np.array(sDOSSet)
        pDOSSetSum += np.array(pDOSSet)
        dDOSSetSum += np.array(dDOSSet)
    # print numRow
    sDOSSet = sDOSSetSum
    pDOSSet = pDOSSetSum
    dDOSSet = dDOSSetSum
   
    eSet   =np.array(eSet   )
    return [eSet,tDOSSet,sDOSSet,pDOSSet,dDOSSet]


def writePOSCAR(output, latConst, latticeVecs, atomSetDirect,\
                comment=None, lSelective=False, lDirect=True):
    """write POSCAR
    defalut commnet is system
    """

    comment = comment or 'system'

    species = [atom[0] for atom in atomSetDirect]
    species1 = list(set(species))
    species1.sort(key=species.index)
    species=species1

    f = open(output,'w')
    # print 'comment', comment
    f.write('{}\n'.format(comment))
    f.write(str(latConst)+'\n')
    

    for latticeVec in latticeVecs:
        for i in range(3):
            f.write(str(latticeVec[i]) + ' ')
        f.write('\n')
    
    for s in species:
        f.write(s + ' ')    
    f.write('\n')

    for s in species:
        n = len([atom for atom in atomSetDirect if atom[0]==s])
        f.write(str(n)+' ' )
    f.write('\n')

    if lSelective: f.write('Selective dynamics\n')
    if lDirect: f.write('Direct\n')
    else : f.write('Cartesian\n')
    for atom in atomSetDirect:
        f.write(str(atom[1][0])+' '+str(atom[1][1])+' '+ str(atom[1][2])+' ' )
        if lSelective:
            # print lSelective
            f.write(str(atom[2][0])+' '+str(atom[2][1])+' '+ str(atom[2][2])+' ' )
        f.write('\n')

    f.close()


def getNELECT(OUTCAR):
    '   NELECT =     338.0000    total number of electrons'
    nelect = findReg('(?<=NELECT =)\s+\d+', OUTCAR)
    return int(nelect[0])


def getVBM(EIGENVAL,band_no = 0):
    if band_no == 0:
        band_no = getNELECT('OUTCAR')/2
    eigenval = findReg('(?<=^'+ str(band_no).rjust(4) +')\s+[-]?[0-9]*\.?[0-9]+',EIGENVAL)
    vbm = [float(e) for e in eigenval]
    vbm = max(vbm)
    return vbm


def getCBM(EIGENVAL,band_no = 0):
    if band_no == 0:
        band_no = getNELECT('OUTCAR') / 2 + 1
    eigenval = findReg('(?<=^'+ str(band_no).rjust(4) +')\s+[-]?[0-9]*\.?[0-9]+',EIGENVAL)
    cbm = [float(e) for e in eigenval]
    cbm = min(cbm)
    return cbm


def getEgap(OUTCAR, EIGENVAL):
    n_elect = getNELECT(OUTCAR)
    vbm = getVBM(EIGENVAL, n_elect/2)
    cbm = getCBM(EIGENVAL, n_elect/2+1)
    egap = cbm - vbm
    return egap


def getTotE(OSZICAR):
    ' 1 F= -.54010511E+03 E0= -.54010511E+03  d E =-.786584E-14'
    energy = findReg('(?<=F=)\s+[-+]?[0-9]*\.?[0-9]+[eE][-+]?[0-9]+? ', OSZICAR)
    totE = [float(e) for e in energy]
    totE = totE[len(totE)-1]
    return totE


def get_tot_E_outcar(outcar, enthalpy=None):
    # get total energy from line of outcar
    '  free  energy   TOTEN  =       -53.472728 eV'
    '''
    FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
    ---------------------------------------------------
    free  energy   TOTEN  =      -178.22800012 eV

    energy  without entropy=     -178.22769568  energy(sigma->0) =     -178.22789864
    enthalpy is  TOTEN    =      -153.45223930 eV   P V=       24.77576082
    '''
    if not enthalpy:
        for line in outcar:
            enthalpy = 'enthalpy' in line
            if enthalpy:
                break
    if enthalpy:
        reg = '(?<=enthalpy is  TOTEN)\s*=\s+[-+]?[0-9]*\.?[0-9]+'
    else:
        reg = '(?<=free  energy   TOTEN)\s*=\s+[-+]?[0-9]*\.?[0-9]+'
    find = []
    for line in outcar:
        m = re.findall(reg, line)
        find += m
    tot_E = float(find[-1].replace('=',''))
    return tot_E


def readOUTCAR(OUTCAR='OUTCAR'):
    f = open(OUTCAR, 'r')
    outcar = f.readlines()
    f.close()
    return outcar

def writeOUTCAR(outcar, output_file='OUTCAR'):
    f = open(output_file, 'w')
    f.writelines(outcar)
    f.close()

def get_enthalpy(dir_name):
    outcar = readOUTCAR('{}/OUTCAR' % dir_name)
    # get total energy from line of outcar
    '  free  energy   TOTEN  =       -53.472728 eV'
    reg = '(?<=TOTEN  =)\s+[-+]?[0-9]*\.?[0-9]+'
    find = []
    for line in outcar:
        m = re.findall(reg, line)
        find += m
    tot_E = float(find[-1])
    return tot_E

def get_eps(OUTCAR='./OUTCAR'):
    """
    return diagonal component of macroscopic dielectric constant
    """
    reg = '(?<=diag\[e\(oo\)\]=\()\s+([\d\.\-]+\s+){3}'
    try:
        with open(OUTCAR) as outcar:
            data = []
            for line in outcar.readlines():
                m = re.search(reg, line)
                if m is not None:
                    # print m.group(1).split()
                    data += m.group().split()
        eps =[]
        for item in data:
            if '-' not in item:
                eps.append(float(item))
        eps = [item for idx, item in enumerate(eps) if idx % 2 == 0]    
        return eps
    except IOError as e: 
        print ("I/O error({0}): {1}".format(e.errno, e.strerror))
        return None

def readTRANCAR(dir_name, file_name='TRANCAR'):
    """
    read output file of optics with LSEARCH = .TRUE.
    optics > TRANCAR
    return list 
    energy[nv, nc, isp, idir, ik]
    amp[nv, nc, isp, idir, ik]
    kpt[nv, nc, isp, idir, ik]
    """

    with open('{}/{}'.format(dir_name, file_name), 'r') as trancar:
        trancar = trancar.readlines()[4:]

        n_item = (len(trancar) + 1 ) / 3

        'NV =    91 NC =   177 ISP = 1 IDIR = 3 IK =       3:'
        n_vb, n_cb, n_spin, n_dir, n_kpt = \
            [int(item.replace(':','')) 
             for item in trancar[-2].split()[2::3]]

        amp_mat = np.zeros((n_vb, n_cb, n_spin, n_dir, n_kpt))
        energy_mat = np.zeros((n_vb, n_cb, n_spin, n_dir, n_kpt))

        # print n_item

        reg_value = ('E =\s*((?:\d*\.)?\d+) '
            'AMP =\s*((?:\d*\.)?\d+) '
            'K =\s*((?:\d*\.)?\d+), '
            '((?:\d*\.)?\d+), '
            '((?:\d*\.)?\d+)')
        for index, line in enumerate(trancar):
            if index % 3 == 0:
                # print index
                'NV =    91 NC =   177 ISP = 1 IDIR = 3 IK =       3:'
                i_vb, i_cb, i_spin, i_dir, i_kpt = \
                    [int(item.replace(':','')) - 1
                     for item in line.split()[2::3]]
            if index % 3 == 1:
                'E =12.97041 AMP = 0.00070 K = 0.0000, 0.0000, 0.0000'
                'E = 8.21349 AMP = 0.00006 K = 0.0000, 0.0000, 0.5000'
                match = re.findall(reg_value, line)
                # print match[0]
                # break
                energy, amp, g_x, g_y, g_z = \
                    [float(item) for item in match[0]]

                amp_mat[i_vb, i_cb, i_spin, i_dir, i_kpt] = amp
                energy_mat[i_vb, i_cb, i_spin, i_dir, i_kpt] = energy

        return energy_mat, amp_mat

def readJDOS(dir_name, JDOS='JDOS'):
    """
    read JDOS file created from
    VASP post-processing optics utility
    return numpy array 
    [energy, jdos, dos]
    dos is shifted in energy
    """
    path = '{}/{}'.format(dir_name, JDOS)
    with open(path) as jdos:
        # ignore first line
        # Joint DOS and DOS shifted by -12.0362803344361531     NEDOS =     4001
        next(jdos)
        jdos = np.loadtxt(jdos)

    return jdos

def read_stress(dir_name, filre_name='OUTCAR'):
    """read stress tensor in OUTCAR
    """
    # read ISIF
    '   ISIF   =      2    stress and relaxation'
    isif_reg = '^\s*ISIF\s+=\s+([1-7])'
    # isif_reg = 'ISIF'
    isif = findReg(isif_reg, '{}/OUTCAR'.format(dir_name))
    isif = int(isif[0])

    # read stress tensor
    """
    FORCE on cell =-STRESS in cart. coord.  units (eV):
    Direction    XX          YY          ZZ          XY          YZ          ZX
    --------------------------------------------------------------------------------------
    Alpha Z     3.36415     3.36415     3.36415
    Ewald     -76.18788   -76.18788   -76.18788    -0.00000     0.00000     0.00000
    Hartree     5.03138     5.03138     5.03138    -0.00000    -0.00000    -0.00000
    E(xc)     -25.40792   -25.40792   -25.40792     0.00001     0.00001     0.00001
    Local     -29.36495   -29.36495   -29.36495     0.00005     0.00005     0.00005
    n-local    75.21641    77.51920    79.63637    -0.69300    -1.40484    -1.64681
    augment   -11.45004   -11.45005   -11.45004    -0.00004    -0.00004    -0.00004
    Kinetic    55.50590    58.10974    56.50063    -1.24871    -2.31794    -2.14173
    Fock        0.00000     0.00000     0.00000     0.00000     0.00000     0.00000
    -------------------------------------------------------------------------------------
    Total       0.14749     0.14749     0.14749     0.00000    -0.00000     0.00000
    in kB       5.90387     5.90387     5.90387     0.00000    -0.00000     0.00000
    external pressure =        5.90 kB  Pullay stress =        0.00 kB
    """
    outcar_file = open('{}/OUTCAR'.format(dir_name))
    outcar = outcar_file.readlines()
    outcar_file.close()
    tensor = np.zeros(6)
    for line_num, line in enumerate(outcar):
        if 'Direction    XX          YY          ZZ'\
              '          XY          YZ          ZX' in line:
            tensor = [float(item) for item in outcar[line_num + 12].split()[1:]]
    metric = np.diag(tensor[:3])
    # metric = tensor[]
    metric[0, 1] = metric[1, 0] = tensor[3]
    metric[1, 2] = metric[2, 1] = tensor[4]
    metric[0, 2] = metric[2, 0] = tensor[5]

    return metric


if __name__ == '__main__':
    path = '/home/users/nwan/asapy/run_subst_100/00000/'
    procar_11 = '/home/users/nwan/cj01/00_NW/R3/00_pristine/unfold/1U_40'
    procar_12 = '/home/users/nwan/cj01/00_NW/R3/00_pristine/unfold/1U_20'
    # energy_mat, amp_mat = readTRANCAR(path)
    # readJDOS(path)
    # readPROCAR(procar_11 + '/PROCAR')
    # readPROCAR_phase(procar_12 + '/PROCAR')
    # readPRJCAR('/home/users/nwan/cj01/00_NW/R3/00_pristine/unfold/bulk/00_222/PRJCAR')
    stress = read_stress('./')
    print (stress)
