import openmm
import numpy as np
import pynmrstar

from openmmnapshift.napshiftforce import NapShiftForce
from pycamcoil.camcoil_engine import CamCoil

ATOM_TYPES = ['CA','CB','C','H','HA','N']
RESIDUE_TYPES = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'CYO':'X', 
                 'GLN':'Q', 'GLU':'E', 'GLY':'G', 'HIS':'H', 'ILE':'I', 
                 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'PRC':'O', 
                 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
CHI1_ATOMS = { "ARG" : ['N', 'CA', 'CB', 'CG'],
               "ASN" : ['N', 'CA', 'CB', 'CG'],
               "ASP" : ['N', 'CA', 'CB', 'CG'],
               "CYS" : ['N', 'CA', 'CB', 'SG'],
               "GLN" : ['N', 'CA', 'CB', 'CG'],
               "GLU" : ['N', 'CA', 'CB', 'CG'],
               "HIS" : ['N', 'CA', 'CB', 'CG'],
               "ILE" : ['N', 'CA', 'CB', 'CG1'],
               "LEU" : ['N', 'CA', 'CB', 'CG'],
               "LYS" : ['N', 'CA', 'CB', 'CG'],
               "MET" : ['N', 'CA', 'CB', 'CG'],
               "PHE" : ['N', 'CA', 'CB', 'CG'],
               "PRO" : ['N', 'CA', 'CB', 'CG'],
               "SER" : ['N', 'CA', 'CB', 'OG'],
               "THR" : ['N', 'CA', 'CB', 'OG1'],
               "TRP" : ['N', 'CA', 'CB', 'CG'],
               "TYR" : ['N', 'CA', 'CB', 'CG'],
               "VAL" : ['N', 'CA', 'CB', 'CG1'] }

CHI2_ATOMS = { "ARG" : ['CA', 'CB', 'CG', 'CD'],
               "ASN" : ['CA', 'CB', 'CG', 'OD1'],
               "ASP" : ['CA', 'CB', 'CG', 'OD1'],
               "GLN" : ['CA', 'CB', 'CG', 'CD'],
               "GLU" : ['CA', 'CB', 'CG', 'CD'],
               "HIS" : ['CA', 'CB', 'CG', 'ND1'],
               "ILE" : ['CA', 'CB', 'CG1', 'CD1'],
               "LEU" : ['CA', 'CB', 'CG', 'CD1'],
               "LYS" : ['CA', 'CB', 'CG', 'CD'],
               "MET" : ['CA', 'CB', 'CG', 'SD'],
               "PHE" : ['CA', 'CB', 'CG', 'CD1'],
               "PRO" : ['CA', 'CB', 'CG', 'CD'],
               "TRP" : ['CA', 'CB', 'CG', 'CD1'],
               "TYR" : ['CA', 'CB', 'CG', 'CD1'] }

def parse_BMRB_entry(BMRB_id, output_dir):
    BMRB_entry = pynmrstar.Entry.from_database(BMRB_id)
    for i, chemical_shit_loop in enumerate(BMRB_entry.get_loops_by_category("Atom_chem_shift")):
        chemical_shifts_data = {}
        for (resid,restype,chainid,atomid,cs_val) in chemical_shit_loop.get_tag(['Comp_index_ID', 'Comp_ID', 'Auth_asym_ID', 'Atom_ID', 'Val']):
            if (resid,restype,chainid) not in chemical_shifts_data.keys(): chemical_shifts_data[(resid,restype,chainid)] = {}
            if atomid == 'H': atomid = 'HN'
            if atomid == 'C': atomid = 'CO'
            chemical_shifts_data[(resid,restype,chainid)][atomid] = cs_val
        with open(f'{output_dir}/{BMRB_id}_CS_{i}.txt', 'w') as f:
            suffix = "_fac"
            f.write(f"{'#NUM' :<8}{'AA' :<8}{''.join([f'{atom:<8}{atom+suffix:<8}' for atom in ATOM_TYPES])}{'CHAIN':<8}\n")
            for (resid,restype,chainid), residue_cs_data in chemical_shifts_data.items():
                f.write(f'{resid:<8}{RESIDUE_TYPES[restype]:<8}')
                for atom in ATOM_TYPES:
                    if atom in residue_cs_data.keys():
                        f.write(f'{residue_cs_data[atom]:<8}{1.0:<8.2f}') 
                    else:
                        f.write(f"{'-':<8}{1.0:<8.2f}") 
                f.write(f'{chainid:<8}\n')

def read_chemical_shifts(chemical_shifts_file):
    chemical_shifts_data = {}
    with open(f'{chemical_shifts_file}', 'r') as f:
        for line in f.readlines()[1:]:
            
            residue_chemical_shift_data = line.split()
            resid = residue_chemical_shift_data[0]
            restype = residue_chemical_shift_data[1]
            chainid = line.split()[-1]
            chemical_shifts = {}
            chemical_shifts_factors = {}
            for atom_index, atom in enumerate(ATOM_TYPES):
                CS = residue_chemical_shift_data[2+atom_index*2]
                chemical_shifts[atom] = float(CS) if CS != '-' else np.nan
                chemical_shifts_factors[atom] = float(residue_chemical_shift_data[2+atom_index*2+1])
            
            chemical_shifts_data[(resid,chainid)] = (restype, chemical_shifts, chemical_shifts_factors)
    return chemical_shifts_data

def get_napshift_force(top, chemical_shifts_file, model_type):
    chemical_shifts_data = read_chemical_shifts(chemical_shifts_file)
    napshiftforce = NapShiftForce()
    camcoil = CamCoil()

    for chain in top.chains():
        # check if this is a protein chain
        if all([residue.name in RESIDUE_TYPES.keys() for residue in chain.residues()]):
            # get protein sequence for this chain, updating CYO and PRC residues according to the chemical shift input file 
            sequence = []
            for residue in chain.residues():
                topology_restype = RESIDUE_TYPES[residue.name]
                if (residue.id,chain.id) in chemical_shifts_data.keys():
                    (restype,_,_) = chemical_shifts_data[(residue.id,chain.id)]
                    if restype == 'X' and topology_restype == 'C': topology_restype = 'X' # the chemical shift file indicates that this CYS residue should be CYO
                    if restype == 'O' and topology_restype == 'P': topology_restype = 'O' # the chemical shift file indicates that this PRO residue should be PRC
                    assert restype == topology_restype 
                sequence.append(topology_restype)
            sequence = ''.join(sequence)
            #predict random coil chemical shifts from this sequence
            camcoil_predictions = camcoil.predict(''.join(sequence))

            for i, residue in enumerate(chain.residues()):
                if residue.name not in RESIDUE_TYPES.keys():continue
                if (residue.id,chain.id) in chemical_shifts_data.keys():
                    restype = sequence[i] # take the residue type from the sequence variable instead of from residue.name, since we may want CYO or PRC instead
                    random_coil_chemical_shifts = {atom: camcoil_predictions.iloc[i][atom] for atom in ATOM_TYPES}
                    experimental_chemical_shifts = chemical_shifts_data[(residue.id,chain.id)][1]
                    experimental_chemical_shift_factors = chemical_shifts_data[(residue.id,chain.id)][2]

                    #get indices of the particles relevant to NapShift for this residue
                    peptide_particle_indices = [-1,-1]
                    if model_type == 'martini':
                        bb_index = int([a.index for a in residue.atoms() if a.name == 'BB'][0])
                        sc_index = int([a.index for a in residue.atoms() if a.name == 'SC1'][0]) if 'SC1' in [a.name for a in residue.atoms()] else -1
                        peptide_particle_indices = [bb_index,sc_index]
                    elif model_type == 'CA':
                        bb_index = int([a.index for a in residue.atoms() if a.name == 'CA'][0])
                        peptide_particle_indices = [bb_index,-1]
                    elif model_type == 'all_atom':
                        N_index = [int(a.index) for a in residue.atoms() if a.name == "N"][0]
                        C_index = [int(a.index) for a in residue.atoms() if a.name == "C"][0]
                        CA_index =[int(a.index) for a in residue.atoms() if a.name == "CA"][0]
                        CB_index =[int(a.index) for a in residue.atoms() if a.name == "CB"][0]
                        G_index = [int(a.index) for a in residue.atoms() if a.name == CHI1_ATOMS[residue.name][-1]][0] if residue.name in CHI1_ATOMS.keys() else -1
                        D_index = [int(a.index) for a in residue.atoms() if a.name == CHI2_ATOMS[residue.name][-1]][0] if residue.name in CHI2_ATOMS.keys() else -1
                        peptide_particle_indices = [N_index,C_index,CA_index,CB_index,G_index,D_index]
                    else:
                        raise NotImplementedError(f"model type {model_type} not implemented")
                    
                    napshiftforce.addPeptide(*peptide_particle_indices,restype,
                                                {k:v if not np.isnan(v) else -1 for k,v in experimental_chemical_shifts.items()}, # -1 to indicate where data is not provided for a chemical shift, and that it should be ignored by the restraints
                                                {k:v if not np.isnan(v) else -1 for k,v in random_coil_chemical_shifts.items()},         # -1 to indicate where data is not provided for a chemical shift, and that it should be ignored by the restraints
                                                experimental_chemical_shift_factors,
                                                int(residue.id),
                                                chain.id)

                                        
    napshiftforce.setModelType(model_type)
    return napshiftforce

def get_restricted_bending_force(top, resids_for_ReB=None):
    restrict_angle_force = openmm.CustomAngleForce("ReB_K/((sin(theta))^2)")
    restrict_angle_force.addGlobalParameter("ReB_K", 0)
    for chain in top.chains():
        if resids_for_ReB is not None:
            CA_atoms = [atom for atom in chain.atoms() if atom.name == "CA" and atom.residue.id in resids_for_ReB]
        else:
            CA_atoms = [atom for atom in chain.atoms() if atom.name == "CA"]            
        for i in range(len(chain)-2):
            restrict_angle_force.addAngle(CA_atoms[i].index, CA_atoms[i+1].index, CA_atoms[i+2].index)
    return restrict_angle_force