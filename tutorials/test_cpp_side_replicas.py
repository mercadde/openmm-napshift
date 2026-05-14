import faulthandler
faulthandler.enable()

import torch

import openmm
from openmm import unit   # openmm = Main OpenMM functionality, unit = Unit/quantity handling
from openmm import app   # app = Application layer (handy interface)
import time
import os
import sys
import numpy as np
import pandas as pd
import argparse
import threading

import MDAnalysis as MDA
from MDAnalysis.lib.util import convert_aa_code

from openmmnapshift.utils import ATOM_TYPES, RESIDUE_TYPES, get_napshift_force

# Disable OpenMM's automatic CUDA Graph capture to prevent cuCtxSynchronize crashes
os.environ["OPENMM_CUDA_DISABLE_GRAPHS"] = "1"

def initProteins():
    proteins = pd.DataFrame(columns=['eps_factor','pH','ionic','fasta'])
    fasta_CTR = """GPKLNLKPRSTPKEDDSSASTSQSTRAASIFGGAKPVDTAAREREVEERLQKEQEKLQRQLDEPKLERRPRERHPSWRSEETQERERSRTGSESSQTGTSTTSSRNARRRESEKSLENETLNKEEDAHSPTSKPPKPDQPLKVMPAPPPKENAWVKRSSNPPARSQSSDTEQQSPTSGGGKVAPAQPSEEGPGRKDENKVDGMNAPKGQTGNSSRGPGDGGNRDHWKESDRKDGKKDQDSRSAPEPKKPEENPASKFSSASKYAALSVDGEDENEGEDYAE""".replace('\n', '')

    proteins.loc['CTR'] = dict(eps_factor=0.2,pH=7.5,fasta=list(fasta_CTR),ionic=0.192)
    return proteins

def genParamsLJ(df,fasta,eps_factor):
    fasta = fasta.copy()
    r = df.copy()
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['X','MW'] += 2
    r.loc['Z','MW'] += 16
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    MWs = [r.loc[a,'MW'] for a in types]
    lj_eps = eps_factor*4.184
    return lj_eps, fasta, types, MWs

def genParamsDH(df,fasta, pH, ionic, temp):
    kT = 8.3145*temp*1e-3
    fasta = fasta.copy()
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H','q'] = 1. / ( 1 + 10**(pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    r.loc['X','q'] = r.loc[fasta[0],'q'] + 1.
    r.loc['Z','q'] = r.loc[fasta[-1],'q'] - 1.
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    #yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    return yukawa_eps#, yukawa_kappa

def add_angles_restriction(top):
    restrict_angle_force = openmm.CustomAngleForce("ReB_K/((sin(theta))^2)")
    restrict_angle_force.addGlobalParameter("ReB_K", 0)
    restrict_angle_force.setUsesPeriodicBoundaryConditions(True)
    exclusions_1_3 = [] # for ah, yu etc.
    for chain in top.chains():
        atoms = [atom for atom in chain.atoms()]
        for i in range(len(chain)-2):
            restrict_angle_force.addAngle(atoms[i].index, atoms[i+1].index, atoms[i+2].index)
            exclusions_1_3.append([atoms[i].index, atoms[i+1].index, atoms[i+2].index])
    return restrict_angle_force, exclusions_1_3

parser = argparse.ArgumentParser()
parser.add_argument('--temp', default=293, type=float, help='simulation temperature')
parser.add_argument('--ionic', default=0.150, type=float, help='simulation ionic strength')
parser.add_argument('--cutoff', default=2.0, type=float, help='cutoff length for nonbonded forces')
parser.add_argument('--GPU', default="0",nargs='?',const='', type=str, help='which GPU to run on (0 or 1)') 

parser.add_argument('--simulation_steps', type=float, default=1e8)
parser.add_argument('--simulation_time', type=float, default=0)
parser.add_argument('--report_interval', type=int, default=1000)

parser.add_argument('--data_dir', default="Data/1DJF", help="top-level directory for the system you want to simulate")
parser.add_argument('--cg_pdb', default="CA.pdb", help="filename for initial structure at <data_dir>/<cg_pdb>")
parser.add_argument('--sim_name', default="test_sim", help="name of the directory to store simulation data in (under <data_dir>/REP_AVG_simulations/<sim_dir>)")

parser.add_argument('--use_NapShift', action=argparse.BooleanOptionalAction)
parser.add_argument('--NapShift_max_K', default=15, type=float)
parser.add_argument('--NapShift_K_gradient', default=0.001, type=float)
parser.add_argument('--CS_filename', default="CS.txt")
parser.add_argument('--num_reps', type=int, default=2)
parser.add_argument('--time_before_warmup', type=float, default=0)
parser.add_argument('--recalculation_interval', type=int, default=1)

parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

def create_simulation(residues,gpu_id,
             temp=293,cutoff=2.0,eps_factor=0.2,ionic=0.192,pH=7.5,
             report_interval=1000,
             sim_name="test_sim", data_dir="EnsembleAvgTestData/eIF4B", cg_pdb="system/original_CA.pdb", CS_filename="NMRData/CS.list", 
             use_NapShift=False, num_reps=1, group_id=9999,
             overwrite=False, minimize=True, add_reporters=True, recalculation_interval=1):
    # create a Simulation object for a single replicate
    
    if add_reporters:
        if not os.path.exists(f"{data_dir}/REP_AVG_simulations/{sim_name}"):
            os.makedirs(f"{data_dir}/REP_AVG_simulations/{sim_name}")
        else:
            if not overwrite and os.path.exists(f"{data_dir}/REP_AVG_simulations/{sim_name}/output.xtc"):
                print(f"error: {data_dir}/REP_AVG_simulations/{sim_name}/output.xtc already exists")
                sys.exit()
        with open(f"{data_dir}/REP_AVG_simulations/{sim_name}/runcommand.txt", "w") as f: f.write(' '.join([str(a) for a in sys.argv]))

    residues = residues.set_index('one')
    lj_eps = eps_factor*4.184
    kT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)

    # set parameters
    L =  100.   
    Lz = 100.

    system = openmm.System()
    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = Lz * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)

    hb = openmm.openmm.HarmonicBondForce()   # implements an interaction between pairs of particles that varies harmonically with the distance between them

    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'

    # Ashbaugh custom NonBonded Force ...
    ah = openmm.openmm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')
    ah.addGlobalParameter('eps',lj_eps*unit.kilojoules_per_mole)
    ah.addGlobalParameter('rc',cutoff*unit.nanometer)
    ah.addPerParticleParameter('s')   # this force depends on two parameters: sigma and lambda ?
    ah.addPerParticleParameter('l')

    # Yukawa custom NonBonded Force ...
    yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addGlobalParameter('shift',np.exp(-yukawa_kappa*4.0)/4.0/unit.nanometer)
    yu.addPerParticleParameter('q')   # another per particle parameter: charge ?

    pdb = app.pdbfile.PDBFile(f'{data_dir}/{cg_pdb}')
    top = pdb.topology
    for j, chain in enumerate(top.chains()):
        fasta = [convert_aa_code(atom.residue.name) for atom in chain.atoms()]
        yukawa_eps = genParamsDH(residues,fasta, pH, ionic, temp)

        atom_indices = [atom.index for atom in chain.atoms()]
        

        system.addParticle((residues.loc[fasta[0]].MW+2)*unit.amu)   # adding chains to system
        for a in fasta[1:-1]:
            system.addParticle(residues.loc[a].MW*unit.amu)
        system.addParticle((residues.loc[fasta[-1]].MW+16)*unit.amu)

        begin = atom_indices[0]
        end = atom_indices[-1]
        for a,e in zip(fasta,yukawa_eps):
            yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])   # call addParticle for each particle in System to define its parameters
            ah.addParticle([residues.loc[a].sigmas*unit.nanometer, residues.loc[a].lambdas*unit.dimensionless])

        for i in range(begin,end):
            hb.addBond(i, i+1, 0.38*unit.nanometer, 8033.0*unit.kilojoules_per_mole/(unit.nanometer**2))   # add a bond term to the forcefield
            yu.addExclusion(i, i+1)   # add a particle pair to the list of interactions that should be excluded
            ah.addExclusion(i, i+1)

    yu.setForceGroup(0)   # set the force group this Force belongs to
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)   # set the method used for handling long range nonbonded interactions
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)   # set PBC
    yu.setCutoffDistance(4*unit.nanometer)   # set the cutoff distance (nm) being used for nonbonded interactions
    ah.setCutoffDistance(cutoff*unit.nanometer)

    num_NapShift_peptides = 0
    if use_NapShift:
        print("Adding NapShift Force")

        napshift_force = get_napshift_force(top, f"{data_dir}/{CS_filename}", model_type='CA')
        napshift_force.setUsesPeriodicBoundaryConditions(True)
        napshift_force.setUsesEnsembleAveraging(True)
        napshift_force.setRecalculationInterval(recalculation_interval)
        napshift_force.setProperty("numReplicas", str(num_reps))
        napshift_force.setProperty("groupId", str(group_id))
        # napshift_force.setProperty("useCUDAGraphs", "false")
        system.addForce(napshift_force)
        num_NapShift_peptides = napshift_force.getNumPeptides()
        
        restricted_angles, _ = add_angles_restriction(top)
        system.addForce(restricted_angles)
        
    system.addForce(hb) 
    system.addForce(yu)
    system.addForce(ah)

    timestep = 0.01*unit.picosecond
    integrator = openmm.openmm.LangevinIntegrator(temp*unit.kelvin,0.01/unit.picosecond,timestep)   # Integrator which simulates using Langevin dynamics

    platform = openmm.Platform.getPlatformByName('CUDA')  
    simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(CudaPrecision='mixed', DeviceIndex=gpu_id))
    
    if minimize:
        simulation.context.setPositions(pdb.positions)  
        simulation.context.setVelocitiesToTemperature(temp*unit.kelvin)
        simulation.minimizeEnergy()  
        
    if add_reporters:
        simulation.reporters.append(app.XTCReporter(f"{data_dir}/REP_AVG_simulations/{sim_name}/output.xtc", report_interval, enforcePeriodicBox=True))
        simulation.reporters.append(app.StateDataReporter(sys.stdout, report_interval, step=True, time=True, 
                                                                potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                                                temperature=True, volume=True, speed=True))
        simulation.reporters.append(app.StateDataReporter(f"{data_dir}/REP_AVG_simulations/{sim_name}/sim.log", report_interval, step=True, time=True, 
                                                                potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                                                temperature=True, volume=True, speed=True))
        simulation.reporters.append(app.CheckpointReporter(f"{data_dir}/REP_AVG_simulations/{sim_name}/checkpoint.chk", report_interval))

    return simulation, num_NapShift_peptides
    
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

def simulate(reps, use_NapShift=False, time_before_warmup=0, timestep=0.01*unit.picosecond,
             NapShift_K_gradient=0, NapShift_max_K=0,
             simulation_steps=0, simulation_time=0):
    if use_NapShift:
        print(f"Pre-warmup")
        steps_before_warmup = int((time_before_warmup * unit.nanosecond) / timestep)
        if steps_before_warmup > 0:
            for rep in reps:
                rep.context.setParameter('NapShift_K', 0)
                rep.context.setParameter('ReB_K', 0)
            step_all(reps, steps_before_warmup)
        
        warmup_steps = int(np.floor((NapShift_max_K - 0)/max(1e-7, NapShift_K_gradient)))
        steps_per_chunk = 100 
        num_chunks = warmup_steps // steps_per_chunk

        print(f"Warming up NapShift for {warmup_steps} steps (in chunks of {steps_per_chunk})...")
        for chunk in range(num_chunks):
            current_i = chunk * steps_per_chunk
            for rep in reps: 
                rep.context.setParameter('NapShift_K', (current_i*NapShift_K_gradient))
                rep.context.setParameter('ReB_K', (current_i*(1/warmup_steps)))
            step_all(reps, steps_per_chunk)
            
        for rep in reps: rep.context.setParameter('NapShift_K', NapShift_max_K)
        for rep in reps: rep.context.setParameter('ReB_K', 1)

    steps = int((simulation_time * unit.nanosecond) / timestep) if simulation_time > 0 else int(simulation_steps)
    print(f"simulating for {steps} steps natively batched on GPU...")
    step_all(reps, steps)

residues = pd.read_csv('Data/CALVADOS_parameters.csv').set_index('three',drop=False)
residues = residues.astype({"q": "float64"})

print("Pre-minimizing starting structure...")
dummy_sim, _ = create_simulation(residues, args.GPU,
                           temp=args.temp,cutoff=args.cutoff,eps_factor=0.2,ionic=args.ionic,pH=7.5,
                           report_interval=args.report_interval,
                           sim_name=args.sim_name, data_dir=args.data_dir, cg_pdb=args.cg_pdb, CS_filename=args.CS_filename, 
                           use_NapShift=False, num_reps=1, group_id=9999,
                           overwrite=args.overwrite, minimize=True, add_reporters=False)
minimized_positions = dummy_sim.context.getState(getPositions=True).getPositions()
del dummy_sim

reps = []
group_id = 9999
for i in range(args.num_reps):
    rep, _ = create_simulation(residues, args.GPU,
                               temp=args.temp,cutoff=args.cutoff,eps_factor=0.2,ionic=args.ionic,pH=7.5,
                               report_interval=args.report_interval,
                               sim_name=f"{args.sim_name}/replicate{i}", data_dir=args.data_dir, cg_pdb=args.cg_pdb, CS_filename=args.CS_filename, 
                               use_NapShift=args.use_NapShift, num_reps=args.num_reps, group_id=group_id,
                               overwrite=args.overwrite, minimize=False, add_reporters=True, recalculation_interval=args.recalculation_interval)
    reps.append(rep)

def init_sim_context(sim_idx, sim):
    sim.context.setPositions(minimized_positions)
    sim.context.setVelocitiesToTemperature(args.temp*unit.kelvin)
    sim.context.getState(getEnergy=True, getForces=True)

print("\n--- Threading Context Initialization ---", flush=True)
init_threads = [threading.Thread(target=init_sim_context, args=(i, s)) for i, s in enumerate(reps)]
for t in init_threads: t.start()
for t in init_threads: t.join()
print("--- Context Initialization Complete ---\n", flush=True)

simulate(reps, use_NapShift=args.use_NapShift, time_before_warmup=args.time_before_warmup,
         NapShift_K_gradient=args.NapShift_K_gradient, NapShift_max_K=args.NapShift_max_K,
         simulation_steps=args.simulation_steps, simulation_time=args.simulation_time)
