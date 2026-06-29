import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import openmm as mm
from openmm import app, unit

import numpy as np
import pandas as pd
import sys
import os
import shutil
import argparse
import threading

from openmmnapshift.utils import get_napshift_force, RESIDUE_TYPES
from openmmnoe.utils import get_NOE_force

def create_replicate(grofile, topfile, simulation_outdir,
                     temperature=298*unit.kelvin,pressure=1,timestep=2*unit.femtosecond, collision_frequency=1/unit.picosecond, GPU=0,
                     simulation_steps=1e8, report_interval=1000,
                     use_NapShift=False, CS_file='TestSystems/FCP1/NMRData/experimental_CS.txt', num_replicates=1, group_id=9999, use_replica_averaging=True, use_CUDA_graphs=False,
                     use_NOEs=False, NOE_file=None, NOE_K=25,
                    ):
    assert not os.path.exists(simulation_outdir)
    os.makedirs(simulation_outdir)
    
    gro = app.GromacsGroFile(grofile)
    gromacs_top = app.GromacsTopFile(topfile, periodicBoxVectors=gro.getPeriodicBoxVectors(),
            includeDir=f'TestSystems/ForceFieldDefinitions/GromacsForceFields/top')
    top = gromacs_top.topology
    system = gromacs_top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer,
            constraints=app.HBonds)

    system.addForce(mm.AndersenThermostat(temperature, collision_frequency))
    system.addForce(mm.MonteCarloBarostat(pressure, temperature))

    if use_NapShift:
        napshift_force = get_napshift_force(top, CS_file, model_type='all_atom')
        napshift_force.setUsesPeriodicBoundaryConditions(True)
        if not use_CUDA_graphs: napshift_force.setProperty("useCUDAGraphs", "false")
        napshift_force.setUsesEnsembleAveraging(use_replica_averaging)
        napshift_force.setProperty("numReplicas", str(num_replicates))
        napshift_force.setProperty("groupId", str(group_id))
        system.addForce(napshift_force)

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

    integrator = mm.VerletIntegrator(timestep)
    platform = mm.Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(top, system, integrator, platform, {"Precision" : "mixed", 'DeviceIndex' : f"{GPU}"})
    simulation.context.setPositions(gro.getPositions())
    print(" --- Energy Minimization --- ")
    state = simulation.context.getState(getEnergy=True)
    print(f"Initial energy: {state.getPotentialEnergy()}")
    simulation.minimizeEnergy()
    state = simulation.context.getState(getEnergy=True)
    print(f"Energy after minimization: {state.getPotentialEnergy()}")

    simulation.context.setVelocitiesToTemperature(temperature)

    xtc_reporter = app.XTCReporter(f'{simulation_outdir}/output.xtc', report_interval, append=False, enforcePeriodicBox=True, atomSubset=[atom.index for atom in top.atoms() if atom.residue.name in RESIDUE_TYPES.keys()])
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
parser.add_argument('--pressure', default=1, type=float, help='simulation pressure (bar)')
parser.add_argument('--GPU', default="0",nargs='?',const='', type=str, help='which GPU to run on (0 or 1)') 

parser.add_argument('--simulation_steps', type=float, default=1e8)
parser.add_argument('--simulation_time', type=float, default=None)
parser.add_argument('--report_interval', type=int, default=1000)

parser.add_argument('--grofile', default="TestSystems/FCP1/ForceFields/CHARMM27/system_extended/ions.gro", help="path to initial structure .gro")
parser.add_argument('--topfile', default="TestSystems/FCP1/ForceFields/CHARMM27/system_extended/topol.top", help="path to .top")
parser.add_argument('--simulation_outdir', default="TestSystems/FCP1/ForceFields/CHARMM27/simulations/test_sim", help="path to simulation output directory")

parser.add_argument('--use_NapShift', action=argparse.BooleanOptionalAction)
parser.add_argument('--NapShift_max_K', default=150, type=float)
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

collision_frequency = 1/unit.picosecond
timestep = 2*unit.femtosecond
steps_per_chunk = args.steps_per_chunk 
simulation_steps = args.simulation_steps if args.simulation_time is None else int((args.simulation_time * unit.nanosecond) / timestep)

reps = []
group_id = 9999
for i in range(args.num_replicates):
    rep = create_replicate(args.grofile, args.topfile, f"{args.simulation_outdir}/rep{i}",
                           temperature=args.temp*unit.kelvin,pressure=args.pressure,timestep=timestep, collision_frequency=collision_frequency,
                           simulation_steps=simulation_steps, report_interval=args.report_interval, GPU=args.GPU,
                           use_NapShift=args.use_NapShift, CS_file=args.CS_file, num_replicates=args.num_replicates, group_id=group_id, use_replica_averaging=args.use_replica_averaging, use_CUDA_graphs=args.use_CUDA_graphs,
                           use_NOEs=args.use_NOEs, NOE_file=args.NOE_file, NOE_K=args.NOE_K
                           )
    reps.append(rep)

if args.use_NapShift:
    # Chemical Shift restraints need to be slowly switched on
    for rep in reps: rep.context.setParameter('NapShift_K', 0)   
    
    warmup_steps = int(np.floor(args.NapShift_max_K/args.NapShift_K_gradient))
    num_chunks = warmup_steps // steps_per_chunk
    
    print(f"Warming up CS restraints for {len(range(warmup_steps))} steps")
    for chunk in range(num_chunks):
        current_step = chunk * steps_per_chunk
        for rep in reps:
            rep.context.setParameter('NapShift_K', (current_step*args.NapShift_K_gradient))
        step_all(reps, steps_per_chunk)
    
    for rep in reps: rep.context.setParameter('NapShift_K', args.NapShift_max_K)  

print(f"Simulating for {simulation_steps} steps")
step_all(reps, simulation_steps)