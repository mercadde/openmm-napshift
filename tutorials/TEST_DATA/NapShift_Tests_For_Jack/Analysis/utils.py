import numpy as np
import MDAnalysis as MDA
import torch
import site

from openmmnapshift.utils import read_chemical_shifts, CHI1_ATOMS, CHI2_ATOMS
from pycamcoil.camcoil_engine import CamCoil

ATOM_TYPES = ["N", "C", "CA", "CB", "H", "HA"]

BLOSUM = {
    "ALA": (+4, -1, -2, -2, +0, +0, -1, -1, +0, -2, -1, -1, -1, -1, -2, -1, -1, +1, +0, -3, -2, +0),
    "ARG": (-1, +5, +0, -2, -3, -3, +1, +0, -2, +0, -3, -2, +2, -1, -3, -2, -2, -1, -1, -3, -2, -3),
    "ASN": (-2, +0, +6, +1, -3, -3, +0, +0, +0, +1, -3, -3, +0, -2, -3, -2, -2, +1, +0, -4, -2, -3),
    "ASP": (-2, -2, +1, +6, -3, -3, +0, +2, -1, -1, -3, -4, -1, -3, -3, -1, -1, +0, -1, -4, -3, -3),
    "CYO": (+0, -3, -3, -3, +9, +8, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -3, -1, -1, -2, -2, -1),
    "CYS": (+0, -3, -3, -3, +8, +9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -3, -1, -1, -2, -2, -1),
    "GLN": (-1, +1, +0, +0, -3, -3, +5, +2, -2, +0, -3, -2, +1, +0, -3, -1, -1, +0, -1, -2, -1, -2),
    "GLU": (-1, +0, +0, +2, -4, -4, +2, +5, -2, +0, -3, -3, +1, -2, -3, -1, -1, +0, -1, -3, -2, -2),
    "GLY": (+0, -2, +0, -1, -3, -3, -2, -2, +6, -2, -4, -4, -2, -3, -3, -2, -2, +0, -2, -2, -3, -3),
    "HIS": (-2, +0, +1, -1, -3, -3, +0, +0, -2, +8, -3, -3, -1, -2, -1, -2, -2, -1, -2, -2, +2, -3),
    "ILE": (-1, -3, -3, -3, -1, -1, -3, -3, -4, -3, +4, +2, -3, +1, +0, -3, -3, -2, -1, -3, -1, +3),
    "LEU": (-1, -2, -3, -4, -1, -1, -2, -3, -4, -3, +2, +4, -2, +2, +0, -3, -3, -2, -1, -2, -1, +1),
    "LYS": (-1, +2, +0, -1, -3, -3, +1, +1, -2, -1, -3, -2, +5, -1, -3, -1, -1, +0, -1, -3, -2, -2),
    "MET": (-1, -1, -2, -3, -1, -1, +0, -2, -3, -2, +1, +2, -1, +5, +0, -2, -2, -1, -1, -1, -1, +1),
    "PHE": (-2, -3, -3, -3, -2, -2, -3, -3, -3, -1, +0, +0, -3, +0, +6, -4, -4, -2, -2, +1, +3, -1),
    "PRC": (-1, -2, -2, -1, -3, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, +7, +6, -1, -1, -4, -3, -2),
    "PRO": (-1, -2, -2, -1, -3, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, +6, +7, -1, -1, -4, -3, -2),
    "SER": (+1, -1, +1, +0, -1, -1, +0, +0, +0, -1, -2, -2, +0, -1, -2, -1, -1, +4, +1, -3, -2, -2),
    "THR": (+0, -1, +0, -1, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, -1, +1, +5, -2, -2, +0),
    "TRP": (-3, -3, -4, -4, -2, -2, -2, -3, -2, -2, -3, -2, -3, -1, +1, -4, -4, -3, -2, 11, +2, -3),
    "TYR": (-2, -2, -2, -3, -2, -2, -1, -2, -3, +2, -1, -1, -2, -1, +3, -3, -3, -2, -2, +2, +7, -1),
    "VAL": (+0, -3, -3, -3, -1, -1, -2, -2, -3, -3, +3, +1, -2, +1, -1, -2, -2, -2, +0, -3, -1, +4),
    "-"  : (+0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0)
}

def dihedral(p):
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return (np.arctan2(y, x))

def angle(p):
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]

    r1 = p0 - p1
    r2 = p2 - p1

    L1 = np.linalg.norm(r1)
    L2 = np.linalg.norm(r2)
    p = np.dot(r1, r2)
    d = p/(L1*L2)

    return np.arccos(d)

class CGPeptide():
    def __init__(self, u, chainID, l2=None, l1=None, c=None, r1=None, r2=None, martiniFormat=False, include_beta=True): #l2-r2 are MDA residues
        c = self.validate_residue(c, chainID, u.residues)
        l2 = self.validate_residue(l2, chainID, u.residues)
        l1 = self.validate_residue(l1, chainID, u.residues)
        r1 = self.validate_residue(r1, chainID, u.residues)
        r2 = self.validate_residue(r2, chainID, u.residues)

        
        self.l2 = l2.atoms[0] if l2 is not None else None
        self.l1 = l1.atoms[0] if l1 is not None else None
        self.c = c.atoms[0] if c is not None else None
        self.r1 = r1.atoms[0] if r1 is not None else None
        self.r2 = r2.atoms[0] if r2 is not None else None
        self.sc = None
        self.martiniFormat = martiniFormat
        self.include_beta = include_beta
        if martiniFormat:
            if self.c is not None:
                self.sc = c.atoms[1] if len(c.atoms) > 1 else None

        self.restype = c.resname if c is not None else "-"

    def validate_residue(self, resnum, chainID, all_residues):
        if resnum < 0 or resnum >= len(all_residues): return None
        if all_residues[resnum].segid != chainID: return None
        return all_residues[resnum]

    def getInputVec(self):
        blosum_vec = BLOSUM[self.restype]
        left_dihed = self.getDihedralInput(self.l2, self.l1, self.c, self.r1)
        right_dihed = self.getDihedralInput(self.l1, self.c, self.r1, self.r2)
        if self.martiniFormat:
            alpha = self.getAngleInput(self.l1, self.c, self.sc)
            beta = self.getAngleInput(self.r1, self.c, self.l1)
            gamma = self.getAngleInput(self.sc, self.c, self.r1)
            inputvec = np.concatenate((blosum_vec, left_dihed, right_dihed, alpha, beta, gamma))
        else:
            if self.include_beta:
                beta = self.getAngleInput(self.r1, self.c, self.l1)
                inputvec = np.concatenate((blosum_vec, left_dihed, right_dihed, beta))
            else:
                inputvec = np.concatenate((blosum_vec, left_dihed, right_dihed))
        return inputvec
    
    def getDihedralInput(self, p1, p2, p3, p4):
        if any(p is None for p in [p1, p2, p3, p4]):
            return np.array((0, 0))
        dihedral_val = dihedral(np.array((p1.position, p2.position, p3.position, p4.position)))
        return(np.array((np.sin(dihedral_val), np.cos(dihedral_val))))
    
    def getAngleInput(self, p1, p2, p3):
        if any(p is None for p in [p1, p2, p3]):
            return np.array((0, 0))
        angle_val = angle(np.array((p1.position, p2.position, p3.position)))
        return(np.array((np.sin(angle_val), np.cos(angle_val))))

class AllAtomPeptide():
    def __init__(self, residue, u):
        
        residue = self.validate_residue(residue, u.residues)

        self.phis, self.psis, self.chi1s, self.chi2s = [], [], [], []
        phi_select, psi_select, chi1_select, chi2_select = None, None, None, None
        if residue is not None:
            phi_select = residue.phi_selection()
            psi_select = residue.psi_selection() 
            if residue.resname in CHI1_ATOMS.keys():
                chi1_select = u.select_atoms(f"resindex {residue.resindex} and name {' '.join(CHI1_ATOMS[residue.resname])}")
                if len(chi1_select) < 4: chi1_select = None
                if chi1_select is not None:
                    chi1_select = sorted([a for a in chi1_select], key=lambda x: CHI1_ATOMS[residue.resname].index(x.name))
                    if [a.name for a in chi1_select] != CHI1_ATOMS[residue.resname]:
                        print("mismatch!!!")
            if residue.resname in CHI2_ATOMS.keys():
                chi2_select = u.select_atoms(f"resindex {residue.resindex} and name {' '.join(CHI2_ATOMS[residue.resname])}")
                if len(chi2_select) < 4: chi2_select = None
                if chi2_select is not None:
                    chi2_select = sorted([a for a in chi2_select], key=lambda x: CHI2_ATOMS[residue.resname].index(x.name))
                    if [a.name for a in chi2_select] != CHI2_ATOMS[residue.resname]:
                        print("mismatch!!!")

        self.phi_select = phi_select
        self.psi_select = psi_select
        self.chi1_select = chi1_select
        self.chi2_select = chi2_select
        self.restype = residue.resname if residue is not None else "-"

    def validate_residue(self, resnum, all_residues):
        if resnum < 0 or resnum >= len(all_residues): return None
        return all_residues[resnum]
    
    def getInputVec(self):
        blosum_vec = BLOSUM[self.restype]
        
        phis = dihedral([p.position for p in self.phi_select]) if self.phi_select is not None else None
        psis = dihedral([p.position for p in self.psi_select]) if self.psi_select is not None else None
        chi1s = dihedral([p.position for p in self.chi1_select]) if self.chi1_select is not None else None
        chi2s = dihedral([p.position for p in self.chi2_select]) if self.chi2_select is not None else None

        phi_inputs = (np.array((np.sin(phis), np.cos(phis)))) if phis is not None else np.array((0,0))
        psi_inputs = (np.array((np.sin(psis), np.cos(psis)))) if psis is not None else np.array((0,0))
        chi1_inputs = (np.array((np.sin(chi1s), np.cos(chi1s)))) if chi1s is not None else np.array((0,0))
        chi2_inputs = (np.array((np.sin(chi2s), np.cos(chi2s)))) if chi2s is not None else np.array((0,0))
        inputvec = np.concatenate((blosum_vec, phi_inputs, psi_inputs, chi1_inputs, chi2_inputs))
        return inputvec

def get_input_peptides(u, mapping="MARTINI3", seq=None):

    if seq is None:
        seq = "".join([MDA.lib.util.convert_aa_code(r.resname) for r in u.residues])

    cc = CamCoil()
    predicted_RC_CS = cc.predict(seq).to_dict(orient='records')

    rc_input = []
    for res in range(len(predicted_RC_CS)):
        rc_input.append([predicted_RC_CS[res][a] for a in ATOM_TYPES])
    random_coil_CS = np.array(rc_input)

    input_peptides = []
    for resnum in range(len(u.residues)):
        if mapping == "MARTINI3":
            chainID = u.residues[resnum].segid
            left_peptide = CGPeptide(u, chainID, l2=resnum-3,l1 =resnum-2,c=resnum-1, r1=resnum, r2=resnum+1, martiniFormat=True)
            center_peptide = CGPeptide(u, chainID, l2=resnum-2,l1 =resnum-1,c=resnum, r1=resnum+1, r2=resnum+2, martiniFormat=True)
            right_peptide = CGPeptide(u, chainID, l2=resnum-1,l1 =resnum,c=resnum+1, r1=resnum+2, r2=resnum+3, martiniFormat=True)
        elif mapping == "CA":
            chainID = u.residues[resnum].segid
            left_peptide = CGPeptide(u, chainID, l2=resnum-3,l1 =resnum-2,c=resnum-1, r1=resnum, r2=resnum+1, martiniFormat=False)
            center_peptide = CGPeptide(u, chainID, l2=resnum-2,l1 =resnum-1,c=resnum, r1=resnum+1, r2=resnum+2, martiniFormat=False)
            right_peptide = CGPeptide(u, chainID, l2=resnum-1,l1 =resnum,c=resnum+1, r1=resnum+2, r2=resnum+3, martiniFormat=False)
        elif mapping == "AllAtom":
            left_peptide = AllAtomPeptide(resnum-1, u)
            center_peptide = AllAtomPeptide(resnum, u)
            right_peptide = AllAtomPeptide(resnum+1, u)
        input_peptides.append([left_peptide, center_peptide, right_peptide])
    return input_peptides, random_coil_CS

def get_chemical_shifts(pdbfile, xtcfile, csfile, mapping='MARTINI3', step=1):
    if mapping == 'MARTINI3':
        model_location = f"{site.getsitepackages()[0]}/openmmnapshift/PytorchModels/martini.pt"
    elif mapping == 'CA':
        model_location = f"{site.getsitepackages()[0]}/openmmnapshift/PytorchModels/CA.pt"
    elif mapping == 'AllAtom':
        model_location = f"{site.getsitepackages()[0]}/openmmnapshift/PytorchModels/all_atom.pt"
    else:
        raise ValueError(f"unrecognised mapping: {mapping}")
    
    NapShift_predictor = torch.load(model_location, weights_only=False).to("cpu")

    experimental_CS = []
    experimental_CS_dict = read_chemical_shifts(csfile)
    for (resid, chainid), (restype, CS_data, _) in experimental_CS_dict.items():
        experimental_CS.append([CS_data[a] for a in ATOM_TYPES])
    experimental_CS = np.array(experimental_CS).T

    u = MDA.Universe(pdbfile, xtcfile)
    input_peptides, random_coil_CS = get_input_peptides(u, mapping=mapping)
    random_coil_CS = random_coil_CS.T


    ensemble_predicted_CS = [[] for atom in ATOM_TYPES]
    traj = u.trajectory[::step]
    for frame, ts in enumerate(traj):
        #if frame % 100 == 0: print(f"{frame}/{len(traj)} ({100*frame/len(traj):.2f}%)")
        input_vectors = []
        for (left, center, right) in input_peptides:
            input_vectors.append(np.concatenate((left.getInputVec(), center.getInputVec(), right.getInputVec())).tolist())
        input_tensor = torch.Tensor(input_vectors)
        predicted_CS = NapShift_predictor(input_tensor)
        predicted_CS = np.where(np.isnan(random_coil_CS.T), torch.nan, predicted_CS.detach().numpy())
        for i, atom in enumerate(ATOM_TYPES):
            ensemble_predicted_CS[i].append(np.array([prediction[i] for prediction in predicted_CS]))
    ensemble_predicted_CS = np.array(ensemble_predicted_CS)
    
    return experimental_CS, random_coil_CS, ensemble_predicted_CS