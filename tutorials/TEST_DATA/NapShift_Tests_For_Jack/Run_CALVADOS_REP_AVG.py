import openmm as mm
from openmm import app, unit

import numpy as np
import pandas as pd
import sys
import os
import shutil
import argparse
import threading

from openmmnapshift.utils import get_napshift_force, get_restricted_bending_force
from openmmnoe.utils import get_NOE_force

"""
Functions to generate nonbonded interaction parameters, and per-residue parameters were modified from the CALVADOS repository at https://github.com/KULL-Centre/CALVADOS.git
Tesei G & Lindorff-Larsen K, Improved predictions of phase behaviour of intrinsically disordered proteins by tuning the interaction range, Open Res Europe, 2023
"""

def genParamsDH(temp,ionic):
    """ Modified from CALVADOS forcefield: https://github.com/KULL-Centre/CALVADOS.git """

    kT = 8.3145*temp*1e-3
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.02214076*1000/kT
    yukawa_eps = lB*kT
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.02214076/10)
    return yukawa_eps, yukawa_kappa

def get_Ashbaugh_Hatch(lj_eps, cutoff, params, top):
    """ Modified from CALVADOS forcefield: https://github.com/KULL-Centre/CALVADOS.git """

    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = mm.CustomNonbondedForce(energy_expression + '; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')
    ah.addGlobalParameter('eps', lj_eps * unit.kilojoules_per_mole)
    ah.addGlobalParameter('rc', float(cutoff) * unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    for r in top.residues():
        ah.addParticle([params.loc[r.name].sigmas * unit.nanometer, params.loc[r.name]['lambdas'] * unit.dimensionless])
           
    ah.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    ah.setCutoffDistance(cutoff*unit.nanometer)
    ah.setForceGroup(0)
    return ah

def get_Yukawa(yukawa_kappa, yukawa_eps, params, top):
    """ Modified from CALVADOS forcefield: https://github.com/KULL-Centre/CALVADOS.git """

    yu = mm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa', yukawa_kappa / unit.nanometer)
    yu.addGlobalParameter('shift', np.exp(-yukawa_kappa * 4.0) / 4.0 / unit.nanometer)
    yu.addPerParticleParameter('q')

    for r in top.residues():
        yu.addParticle([params.loc[r.name].q*np.sqrt(yukawa_eps) * unit.nanometer * unit.kilojoules_per_mole])

    yu.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    yu.setCutoffDistance(4*unit.nanometer)
    yu.setForceGroup(1)
    return yu

def add_bonds(top, bond_length, k_bond):
    harmonic_bond_force = mm.HarmonicBondForce()
    harmonic_bond_force.setUsesPeriodicBoundaryConditions(True)
    exclusions_1_2 = [] # for ah, yu etc.
    for chain in top.chains():
        atoms = [atom for atom in chain.atoms()]
        for i in range(len(chain)-1):
            harmonic_bond_force.addBond(atoms[i].index, atoms[i+1].index, bond_length*unit.nanometer, k_bond*unit.kilojoules_per_mole/(unit.nanometer**2))
            exclusions_1_2.append((atoms[i].index, atoms[i+1].index))
    return harmonic_bond_force, exclusions_1_2

def create_replicate(cg_pdb, simulation_outdir,
                     temperature=298,salt_conc=0.150,timestep=10*unit.femtosecond, box_len=50.0,
                     simulation_steps=1e8, report_interval=1000, GPU=0,
                     use_NapShift=False, CS_file='TestSystems/FCP1/NMRData/experimental_CS.txt', num_replicates=1, group_id=9999, use_replica_averaging=True, use_CUDA_graphs=False,
                     use_NOEs=False, NOE_file=None, NOE_K=25,
                    ):
    assert not os.path.exists(simulation_outdir)
    os.makedirs(simulation_outdir)
    
    bond_length = 0.38
    k_bond = 8033.0
    eps_lj = 0.2 * 4.184 # kcal to kJ/mol
    cutoff_lj = 2.2
    yukawa_eps, yukawa_kappa = genParamsDH(temperature.value_in_unit(unit.kelvin), salt_conc)
    CALVADOS_parameters = pd.read_csv('TestSystems/ForceFieldDefinitions/CALVADOS_parameters.csv', index_col='three') # from https://github.com/KULL-Centre/CALVADOS/blob/main/residues.csv

    cg_pdb = app.PDBFile(cg_pdb)
    top = cg_pdb.topology

    for i, chain in enumerate(top.chains()):
        chain.id = f"{i+1}"

    system = mm.System()
    for r in top.residues():
        system.addParticle(CALVADOS_parameters.loc[r.name].MW*unit.amu)
    system.setDefaultPeriodicBoxVectors(np.array([box_len,0,0]), np.array([0,box_len,0]), np.array([0,0,box_len]))

    ah = get_Ashbaugh_Hatch(eps_lj, cutoff_lj, CALVADOS_parameters, top)
    yu = get_Yukawa(yukawa_eps, yukawa_kappa, CALVADOS_parameters, top)
    hb, exclusions_1_2 = add_bonds(top, bond_length, k_bond)
    system.addForce(hb)

    for i, j in exclusions_1_2:
        ah.addExclusion(i,j)
        yu.addExclusion(i,j)

    system.addForce(ah)
    system.addForce(yu)

    if use_NapShift:
        napshift_force = get_napshift_force(top, CS_file, model_type='CA')
        napshift_force.setUsesPeriodicBoundaryConditions(True)
        if not use_CUDA_graphs: napshift_force.setProperty("useCUDAGraphs", "false")
        napshift_force.setUsesEnsembleAveraging(use_replica_averaging)
        napshift_force.setProperty("numReplicas", str(num_replicates))
        napshift_force.setProperty("groupId", str(group_id))
        system.addForce(napshift_force)

        # apply a restricted bending force to handle instabilities arrising from appling a dihedral potential in a coarse-grained setting 
        # see M. Bulacu et al., Improved Angle Potentials for Coarse-Grained Molecular Dynamics Simulations, J. Chem. Theory Comput., 2013.
        restricted_bending_force = get_restricted_bending_force(top)
        restricted_bending_force.setUsesPeriodicBoundaryConditions(True)
        system.addForce(restricted_bending_force)

    if use_NOEs:
        noe_force = get_NOE_force(top, NOE_file,
                            excluded_atom_names=[],
                            minimum_residue_seperation=5,
                            secondary_structure_blocks=[],
                            apply_intra_secondary_structre_NOEs=False,
                            apply_inter_chain_NOEs=True)

        noe_force.setUsesPeriodicBoundaryConditions(True)
        noe_force.setK(NOE_K)
        system.addForce(noe_force)


    integrator = mm.LangevinMiddleIntegrator(temperature, 0.01/unit.picosecond, timestep)
    platform = mm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(top, system, integrator, platform, {"Precision" : "mixed", 'DeviceIndex' : f"{GPU}"})
    simulation.context.setPositions(cg_pdb.getPositions())
    print(" --- Energy Minimization --- ")
    state = simulation.context.getState(getEnergy=True)
    print(f"Initial energy: {state.getPotentialEnergy()}")
    simulation.minimizeEnergy()
    state = simulation.context.getState(getEnergy=True)
    print(f"Energy after minimization: {state.getPotentialEnergy()}")
    simulation.context.setVelocitiesToTemperature(temperature)

    xtc_reporter = app.XTCReporter(f'{simulation_outdir}/output.xtc', report_interval, append=False, enforcePeriodicBox=True)
    state_data_reporter_stdout = app.StateDataReporter(sys.stdout, report_interval, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, speed=True, progress=True, remainingTime=True, totalSteps=simulation_steps)
    state_data_reporter_logfile = app.StateDataReporter(f'{simulation_outdir}/sim.log', report_interval, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, progress=True, remainingTime=True, speed=True, elapsedTime=True, systemMass=True, totalSteps=simulation_steps)
    simulation.reporters.append(xtc_reporter)
    simulation.reporters.append(state_data_reporter_stdout)
    simulation.reporters.append(state_data_reporter_logfile)

    return simulation

def safe_step(sim_idx, sim, num_steps):
    try:
        sim.step(num_steps)
    except Exception as e:
        print(f"\n[FATAL ERROR IN THREAD] OpenMM Simulation threw an exception:\n{e}", file=sys.stderr)
        os._exit(1)

def step_all(reps, num_steps):
    if num_steps <= 0: return
    threads = [threading.Thread(target=safe_step, args=(i, s, num_steps)) for i, s in enumerate(reps)]
    for t in threads: t.start()
    for t in threads: t.join()

parser = argparse.ArgumentParser()
parser.add_argument('--temp', default=293, type=float, help='simulation temperature (K)')
parser.add_argument('--ionic', default=0.150, type=float, help='simulation ionic strength (M)')
parser.add_argument('--box_len', default=100.0, type=float, help='side length for periodic box')
parser.add_argument('--GPU', default="0",nargs='?',const='', type=str, help='which GPU to run on (0 or 1)') 

parser.add_argument('--simulation_steps', type=float, default=1e8)
parser.add_argument('--simulation_time', type=float, default=None)
parser.add_argument('--report_interval', type=int, default=1000)

parser.add_argument('--cg_pdb', default="TestSystems/FCP1/ForceFields/CALVADOS/system/helical_modelled_CA.pdb", help="path to initial structure pdb")
parser.add_argument('--simulation_outdir', default="TestSystems/FCP1/ForceFields/CALVADOS/simulations/test_sim", help="path to simulation output directory")

parser.add_argument('--use_NapShift', action=argparse.BooleanOptionalAction)
parser.add_argument('--NapShift_max_K', default=15, type=float)
parser.add_argument('--NapShift_K_gradient', default=0.001, type=float)
parser.add_argument('--CS_file', default="TestSystems/FCP1/NMRData/experimental_CS.txt", help="input Chemical Shift file")
parser.add_argument('--use_replica_averaging', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--use_CUDA_graphs', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--num_replicates', default=1, type=int)
parser.add_argument('--steps_per_chunk', default=1, type=int)

parser.add_argument('--use_NOEs', action=argparse.BooleanOptionalAction)
parser.add_argument('--NOE_K', default=25, type=float)
parser.add_argument('--NOE_file', default=None, help="input NOE file")

parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

simulation_outdir = args.simulation_outdir
if os.path.exists(simulation_outdir):
    if args.overwrite:
        shutil.rmtree(simulation_outdir)
    else:
        raise OSError(f"target simulation outdir {simulation_outdir} already exists!")
os.makedirs(simulation_outdir)

timestep = 10*unit.femtosecond
steps_per_chunk = args.steps_per_chunk 
simulation_steps = args.simulation_steps if args.simulation_time is None else int((args.simulation_time * unit.nanosecond) / timestep)

reps = []
group_id = 9999
for i in range(args.num_replicates):
    rep = create_replicate(args.cg_pdb, f"{args.simulation_outdir}/rep{i}",
                                temperature=args.temp*unit.kelvin,salt_conc=args.ionic,timestep=timestep, box_len=args.box_len,
                                simulation_steps=simulation_steps, report_interval=args.report_interval, GPU=args.GPU,
                                use_NapShift=args.use_NapShift, CS_file=args.CS_file, num_replicates=args.num_replicates, group_id=group_id, use_replica_averaging=args.use_replica_averaging, use_CUDA_graphs=args.use_CUDA_graphs,
                                use_NOEs=args.use_NOEs, NOE_file=args.NOE_file, NOE_K=args.NOE_K
                                )
    reps.append(rep)

if args.use_NapShift:
    # Chemical Shift restraints (and the restricted bending force) need to be slowly switched on
    for rep in reps: rep.context.setParameter('NapShift_K', 0)   
    for rep in reps: rep.context.setParameter('ReB_K', 0)   

    
    warmup_steps = int(np.floor(args.NapShift_max_K/args.NapShift_K_gradient))
    num_chunks = warmup_steps // steps_per_chunk
    
    print(f"Warming up CS restraints for {len(range(warmup_steps))} steps")
    for chunk in range(num_chunks):
        current_step = chunk * steps_per_chunk
        for rep in reps:
            rep.context.setParameter('NapShift_K', (current_step*args.NapShift_K_gradient))
            rep.context.setParameter('ReB_K', (current_step*(1/warmup_steps)))
        step_all(reps, steps_per_chunk)
    
    for rep in reps: rep.context.setParameter('NapShift_K', args.NapShift_max_K)   
    for rep in reps: rep.context.setParameter('ReB_K', 1)   

print(f"Simulating for {simulation_steps} steps")
step_all(reps, simulation_steps)