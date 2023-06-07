import numpy as np
import sys
import joblib
from ase import Atoms
import ase.io
from dscribe.descriptors import CoulombMatrix
from pyscf import gto
from joblib import Parallel, delayed

from subroutine_rotation import get_standard_orientation, rotate_molecular_orbital

'''
FMO Coupling Prediction
@author Hung-Hsuan Lin

Requirements:
    numpy 1.20.3
    mendeleev 0.9.0
    Dscribe 1.2.0
    ase 3.22.1
    pyscf 2.0.1
Changelog:
    ver 1.00    2022 Feb 15
Additional Info:
    Input coordinate using QChem format
'''

def PredictionfromQChem(QChem):
	'''==============================
	0. Read Parameters
	=============================='''
	''' 0.1. Get the GTO basis set
	   =============================='''
	gto_basis = {'C': gto.load('dz_q-chem.dat', 'C'), 'H': gto.load('dz_q-chem.dat', 'H')}
	''' 0.2. Read the Coordinate and MO Coefficients of Reference
	   =============================='''
	Ref = ase.io.read('Eth_sym.SO.xyz')
	RefCoor = Ref.get_positions().reshape(1,-1)
	RefMO = np.fromfile('Eth_sym.SO.out.MO')

	'''==============================
	1. Read Input (QChem Format)
	=============================='''
	''' 1.1. Check if the input file exist
	   =============================='''
	with open(QChem) as inline:
	    coor = []
	    for line in inline:
	        if "$molecule" in line:
	            line = next(inline)
	            line = next(inline)
	            while "$end" not in line:
	                coor.append(line.split())
	                line = next(inline)

	''' 1.2. Get the dimension of the fragment 1 and 2
	   =============================='''
	indices = [i for i, x in enumerate(coor) if x == ['--']]
	NDim_Fragment1 = indices[1] - 2
	NDim_Fragment2 = len(coor) - NDim_Fragment1 - 4
	NDim = NDim_Fragment1 + NDim_Fragment2
	
	''' 1.3. Remove unnecessary elements
	   =============================='''
	coor.remove(['--'])
	coor.remove(['--'])
	coor.remove(['0', '1'])
	coor.remove(['0', '1'])

	''' 1.4. Separate the fragments
	   =============================='''
	Fragment1=coor[:NDim_Fragment1]
	Fragment2=coor[NDim_Fragment1:]
	
	''' 1.5. Separate atoms and coordinates
	   =============================='''
	Fragment1_atoms = []
	for i in Fragment1:
	    Fragment1_atoms.append(i[0])
	    for j in range(1,4):
	        i[j] = float(i[j])
	    i.pop(0)
	Fragment1=Atoms(Fragment1_atoms, positions=Fragment1)
	
	Fragment2_atoms = []
	for i in Fragment2:
	    Fragment2_atoms.append(i[0])
	    for j in range(1,4):
	        i[j] = float(i[j])
	    i.pop(0)
	Fragment2=Atoms(Fragment2_atoms, positions=Fragment2)

	Dimer = Fragment1 + Fragment2
	''' 1.6. Create pyscf objects for the fragments and the dimer
	   =============================='''
	Fragment1_atoms, Fragment1_coor = Fragment1.get_chemical_symbols(), Fragment1.get_positions()
	Fragment1_pyscf_mol = list(zip(Fragment1_atoms, Fragment1_coor))
	Fragment1_pyscf_obj = gto.M(atom=Fragment1_pyscf_mol, basis=gto_basis)

	Fragment2_atoms, Fragment2_coor = Fragment2.get_chemical_symbols(), Fragment2.get_positions()
	Fragment2_pyscf_mol = list(zip(Fragment2_atoms, Fragment2_coor))
	Fragment2_pyscf_obj = gto.M(atom=Fragment2_pyscf_mol, basis=gto_basis)

	dimer_pyscf_obj = Fragment1_pyscf_obj+Fragment2_pyscf_obj
	'''==============================================
	2. Get the Dimer in Coulomb Matrix Representation
	=============================================='''
	cm = CoulombMatrix(
	    n_atoms_max=len(Fragment1_atoms)+len(Fragment2_atoms), flatten=False,
	    permutation='none')
	dimer_cm = cm.create(Dimer)
	dimer_cm = dimer_cm[0:NDim_Fragment1, NDim_Fragment1:NDim].reshape(1,-1) ### consider intermolecular block
	
	'''==============================
	3. Rotate the fragments into the Standard Orientation
	=============================='''
	Fragment1_SO, Fragment1_RotationMatrix = get_standard_orientation(Fragment1)
	Fragment2_SO, Fragment2_RotationMatrix = get_standard_orientation(Fragment2)

	'''==============================
	4. Predict MO 
	=============================='''
	''' 4.1. Predict MO in SO 
	   =============================='''
	NDimMO=int(dimer_pyscf_obj.nao/2) ### divided by 2 because of the restricated orbitals
	HOMO=int(dimer_pyscf_obj.nelectron/4 - 1)
	homo_prediction_model = joblib.load('EthyleneDimer_Cart-difftoMO_15k_KRR_final.joblib')
	
	Fragment1_coor_diff = Fragment1_SO - RefCoor
	Fragment1_homo_diff = homo_prediction_model.predict(np.array(Fragment1_coor_diff.reshape(1,-1)))
	Fragment1_homo = Fragment1_homo_diff + RefMO[:NDimMO*NDimMO].reshape(NDimMO,NDimMO)[HOMO,:]
	
	Fragment2_coor_diff = Fragment2_SO - RefCoor
	Fragment2_homo_diff = homo_prediction_model.predict(np.array(Fragment2_coor_diff.reshape(1,-1)))
	Fragment2_homo = Fragment2_homo_diff + RefMO[:NDimMO*NDimMO].reshape(NDimMO,NDimMO)[HOMO,:]
	
	''' 4.2. Rotate MO back
	   =============================='''
	Rotated_Frag1_HOMO = rotate_molecular_orbital(Fragment1_RotationMatrix.flatten(), Fragment1_homo[0], Fragment1_pyscf_obj)
	Rotated_Frag2_HOMO = rotate_molecular_orbital(Fragment2_RotationMatrix.flatten(), Fragment2_homo[0], Fragment2_pyscf_obj)

	'''==============================
	5. Evalaute Overlap
	=============================='''
	''' 5.1. Calculate the Overlap Matrix of the Dimer
	   =============================='''
	OverlapMatrix = dimer_pyscf_obj.intor('int1e_ovlp', hermi=0)
	#OverlapMatrix = OverlapMatrix.reshape(-1,1)
	
	''' 5.2. Calculate the Overlap Trace
	   =============================='''
	jPML=np.outer(Rotated_Frag1_HOMO, Rotated_Frag2_HOMO).T
	overlap = np.trace(np.matmul(jPML,OverlapMatrix[:NDimMO, NDimMO:]))
	dimer_feature = dimer_cm
	
	''' 5.3. Predict the FMO Coupling from Machine Learning Overlap
	   =============================='''
	Feature=np.hstack([dimer_feature, np.array([overlap]).reshape((1,-1))])
	kr=joblib.load('EthyleneDimer_CM_inter_OverlaptoCoupling_40k_KRR.joblib')
	ML_EC=kr.predict(Feature)
	return ML_EC[0]

if __name__ == '__main__':
	ML_EC = PredictionfromQChem(sys.argv[1])
	print(ML_EC)
