import openmm as mm
from openmm import app, unit

from openmm.app.gromacsgrofile import GromacsGroFile
import martini_openmm as martini

import numpy as np
import sys
import os
import shutil
import argparse

from openmmnapshift.utils import get_napshift_force, RESIDUE_TYPES
from openmmnoe.utils import get_NOE_force

parser = argparse.ArgumentParser()
parser.add_argument('--temp', default=293, type=float, help='simulation temperature (K)')
parser.add_argument('--pressure', default=1, type=float, help='simulation pressure (bar)')
parser.add_argument('--GPU', default="0",nargs='?',const='', type=str, help='which GPU to run on (0 or 1)') 

parser.add_argument('--simulation_steps', type=float, default=1e8)
parser.add_argument('--simulation_time', type=float, default=0)
parser.add_argument('--report_interval', type=int, default=1000)

parser.add_argument('--grofile', default="TestSystems/FCP1/ForceFields/Martini3/system_extended/ions.gro", help="path to initial structure .gro")
parser.add_argument('--topfile', default="TestSystems/FCP1/ForceFields/Martini3/system_extended/topol.top", help="path to .top")
parser.add_argument('--simulation_outdir', default="TestSystems/FCP1/ForceFields/Martini3/simulations/test_sim", help="path to simulation output directory")

parser.add_argument('--use_NapShift', action=argparse.BooleanOptionalAction)
parser.add_argument('--NapShift_max_K', default=15, type=float)
parser.add_argument('--NapShift_K_gradient', default=0.001, type=float)
parser.add_argument('--CS_file', default="TestSystems/FCP1/NMRData/experimental_CS.txt", help="input Chemical Shift file")

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

max_K = args.NapShift_max_K
K_gradient = args.NapShift_K_gradient

report_interval = args.report_interval
simulation_steps = args.simulation_time if args.simulation_time > 0 else args.simulation_steps

temperature = args.temp*unit.kelvin
pressure = args.pressure
timestep = 10*unit.femtosecond

conf = GromacsGroFile(args.grofile)
box_vectors = conf.getPeriodicBoxVectors()
martini_topfile = martini.MartiniTopFile(args.topfile, periodicBoxVectors=box_vectors, epsilon_r=15)
top = martini_topfile.topology
system = martini_topfile.create_system(nonbonded_cutoff=1.1 * unit.nanometer)
barostat = mm.MonteCarloBarostat(pressure, temperature)
system.addForce(barostat)

if args.use_NapShift:
    napshift_force = get_napshift_force(top, args.CS_file, model_type='martini')
    napshift_force.setUsesPeriodicBoundaryConditions(True)
    system.addForce(napshift_force)

if args.use_NOEs:
    noe_force = get_NOE_force(top, args.NOE_file,
                          excluded_atom_names=[],
                          minimum_residue_seperation=5,
                          secondary_structure_blocks=[],
                          apply_intra_secondary_structre_NOEs=False,
                          apply_inter_chain_NOEs=True)

    noe_force.setUsesPeriodicBoundaryConditions(True)
    noe_force.setK(args.NOE_K)
    system.addForce(noe_force)

integrator = mm.LangevinMiddleIntegrator(temperature, 0.01/unit.picosecond, timestep)
platform = mm.Platform.getPlatformByName("CUDA")
simulation = app.Simulation(top, system, integrator, platform, {"Precision" : "mixed", 'DeviceIndex' : "0"})
simulation.context.setPositions(conf.getPositions())
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)

xtc_reporter = app.XTCReporter(f'{simulation_outdir}/output.xtc', report_interval, append=False, enforcePeriodicBox=True, atomSubset=[atom.index for atom in top.atoms() if atom.residue.name in RESIDUE_TYPES.keys()])
state_data_reporter_stdout = app.StateDataReporter(sys.stdout, report_interval, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, speed=True, progress=True, remainingTime=True, totalSteps=simulation_steps)
state_data_reporter_logfile = app.StateDataReporter(f'{simulation_outdir}/sim.log', report_interval, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, progress=True, remainingTime=True, speed=True, elapsedTime=True, systemMass=True, totalSteps=simulation_steps)
simulation.reporters.append(xtc_reporter)
simulation.reporters.append(state_data_reporter_stdout)
simulation.reporters.append(state_data_reporter_logfile)

if args.use_NapShift:
    # Chemical Shift restraints need to be slowly switched on
    warmup_steps = int(np.floor(max_K/K_gradient))
    print(f"Warming up CS restraints for {len(range(warmup_steps))} steps")
    for i in range(warmup_steps):
        simulation.step(1)
        simulation.context.setParameter('NapShift_K', (i*K_gradient))

print(f"Simulating with CS restraints")
simulation.step(simulation_steps)