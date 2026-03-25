/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2024 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors: Raimondas Galvelis, Raul P. Pelaez                           *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

//#include <cuda_profiler_api.h>

#include "CudaNapShiftKernels.h"
#include "CudaNapShiftKernelSources.h"

#include "openmm/common/ContextSelector.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/internal/ContextImpl.h"
//#include "openmm/cuda/src/CudaKernelSources.h"

#include <map>
#include <cmath>
#include <memory>
#include <chrono>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/torch.h>
#include <torch/types.h>
#include <torch/csrc/jit/serialization/import.h>
#include <ATen/ATen.h>

using namespace NapShiftPlugin;
using namespace OpenMM;
using namespace std;

// macro for checking the result of synchronization operation on CUDA
// copied from `openmm/platforms/cuda/src/CudaParallelKernels.cpp`
#define CHECK_RESULT(result, prefix)                                             \
    if (result != CUDA_SUCCESS) {                                                \
        std::stringstream m;                                                     \
        m << prefix << ": " << cu.getErrorString(result) << " (" << result << ")"\
          << " at " << __FILE__ << ":" << __LINE__;                              \
        throw OpenMMException(m.str());                                          \
    }

CudaCalcNapShiftForceKernel::CudaCalcNapShiftForceKernel(string name, const Platform& platform, CudaContext& cu) : CalcNapShiftForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    // Explicitly activate the primary context
    CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext, cu.getDevice()), "Failed to retain the primary context");
}

CudaCalcNapShiftForceKernel::~CudaCalcNapShiftForceKernel() {
    cuDevicePrimaryCtxRelease(cu.getDevice());
}

static int round64(int x)
{
    int remainder = x % 64;
    if (remainder == 0)
        return max(x, 64);
    int rounded = x + 64 - remainder;
    return rounded < 64 ? 64 : rounded;
}

/**
 * Get a pointer to the data in a PyTorch tensor.
 * The tensor is converted to the correct data type if necessary.
 */
static void* getTensorPointer(OpenMM::CudaContext& cu, torch::Tensor& tensor) {
    void* data;
    if (tensor.dtype() == torch::kInt) {
        data = tensor.to(torch::kInt).data_ptr<int>();
    }
    else {
        if (cu.getUseDoublePrecision()) {
            data = tensor.to(torch::kFloat64).data_ptr<double>();
        } else {
            data = tensor.to(torch::kFloat32).data_ptr<float>();
        }
    }
    return data;
}

static vector<int> linearize(const std::vector<std::vector<int>>& vec_vec) {
    std::vector<int> vec;
    for (const auto& v : vec_vec) {
        for (auto d : v) {
            vec.push_back(d);
        }
    }
    return vec;
}

void CudaCalcNapShiftForceKernel::initialize(const System& system, const NapShiftForce& force) {
    CUdevice cudevice = cu.getDevice();
    cuDeviceGetAttribute(&device_multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudevice);
    cuDeviceGetAttribute(&device_max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cudevice);
  
    numAtomTypes = atomTypes.size();

    std::map<std::string, float>modelErrorMap;
    std::string modelFilename;
    if (force.getModelType() == "martini") {
        modelFormat = 0;
        numInputAngles = 5;
        modelErrorMap = martiniModelErrorMap;
        modelFilename = force.getPytorchModelsDir()+"/martini.pt";
    } else if (force.getModelType() == "CA") {
        modelFormat = 1;
        numInputAngles = 3;
        modelErrorMap = CAModelErrorMap;
        modelFilename =  force.getPytorchModelsDir()+"/CA.pt";
    }else if (force.getModelType() == "all_atom") {
        modelFormat = 2;
        numInputAngles = 4;
        modelErrorMap = allAtomModelErrorMap;
        modelFilename =  force.getPytorchModelsDir()+"/all_atom.pt";
    } else throw OpenMMException("NapShiftForce: invalid value of \"modelType\"");

    oneInputVecSize = 22 + numInputAngles*2;
    fullInputVecSize  = 3*oneInputVecSize;

    usePeriodic = force.usesPeriodicBoundaryConditions();
    ensembleAveraging = force.usesEnsembleAveraging();
    numPeptides = force.getNumPeptides();
    int numParticles = system.getNumParticles();

    // Push the PyTorch context
    // NOTE: Pytorch is always using the primary context.
    //       It makes the primary context current, if it is not a case.
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

    // Initialize CUDA objects for PyTorch
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex()); // This implicitly initializes PyTorch
    realOptionsDevice = torch::TensorOptions().dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32).device(device);
    intOptionsDevice = torch::TensorOptions().dtype(torch::kInt).device(device);
    realOptionsCPU = torch::TensorOptions().dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32).device("cpu");
    intOptionsCPU = torch::TensorOptions().dtype(torch::kInt).device("cpu");

    model = torch::jit::load(modelFilename, device);
    model.to(device);
    model.eval();
    model = torch::jit::freeze(model);

    modelErrors = torch::empty({numAtomTypes}, realOptionsCPU);
    ChemShiftSTD = torch::empty({numAtomTypes}, realOptionsCPU);
    for (int a=0; a<numAtomTypes; a++) {
        std::string atomType = atomTypes[a];
        modelErrors[a] = modelErrorMap[atomType];
    }

    modelErrors = modelErrors.to(device);
    ChemShiftSTD = ChemShiftSTD.to(device);
    K = torch::zeros({1}, realOptionsDevice);
    RMSD = torch::empty({6}, realOptionsDevice);
    csTensor = torch::empty({numPeptides, numAtomTypes}, realOptionsDevice);
    randomCoilTensor = torch::empty({numPeptides, numAtomTypes}, realOptionsCPU);
    CSExpTensor = torch::empty({numPeptides, numAtomTypes}, realOptionsCPU);
    ChemShiftScale = torch::empty({numPeptides, numAtomTypes}, realOptionsCPU);

    //ensemble averaging
    avgCSTensor = torch::empty({numPeptides, numAtomTypes}, realOptionsDevice);
    dNapShift_dDelta = torch::zeros({numPeptides, numAtomTypes});
    dNapShift_dDelta = dNapShift_dDelta.to(device);
    // ---
    csDifference = torch::empty({numPeptides, numAtomTypes}, realOptionsDevice);

    energyTensor = torch::empty({1}, realOptionsDevice);
    dNN_dAngle = torch::empty({numPeptides, fullInputVecSize}, realOptionsDevice);
    relevant_dNN_dAngle = torch::empty({numPeptides, 3*numInputAngles*2}, realOptionsDevice);
    // ---

    //read NapShift peptides from frontend
    for (int i = 0; i < numPeptides; i++) {
        char resType;
        std::map<std::string, double> csExp;
        std::map<std::string, double> csRC;
        std::map<std::string, double> csScale;
        int resId;
        std::string chainId;
        if (modelFormat == 0 || modelFormat == 1) { //CG format
            int bbIndex, scIndex;   
            force.getPeptideParameters(i, bbIndex, scIndex, resType, csExp, csRC, csScale, resId, chainId);
            peptides.push_back(std::make_unique<Peptide>(bbIndex, scIndex, resType, csExp, csRC, csScale, resId, chainId));
        } else { //all-atom format
            int N, C, CA, CB, G, D;
            force.getPeptideParameters(i, N, C, CA, CB, G, D, resType, csExp, csRC, csScale, resId, chainId);
            peptides.push_back(std::make_unique<AllAtomPeptide>(N, C, CA, CB, G, D, resType, csExp, csRC, csScale, resId, chainId));
        }
    }
    //sort peptides by resid and chain to ensure that we create peptides triplets in the correct order
    std::sort(peptides.begin(),
              peptides.end(),
              [](const std::unique_ptr<BasePeptide>& lhs, const std::unique_ptr<BasePeptide>& rhs)
              {
                  if (lhs->getChainId() != rhs->getChainId()) return lhs->getChainId() < rhs->getChainId();
                  return lhs->getResId() < rhs->getResId();
              });
    //now grow the list of particles from the correctly ordered NapShift peptide list
    for (int i = 0; i < numPeptides; i++) {
        for (int index : peptides[i]->getParticleIndices()){
            if (index > -1) NapShiftParticles.push_back(index);
        }
    }
    
    indexToAtomArray.initialize<int>(cu, numParticles, "NapShiftIndexToAtomArray");

    numNapShiftParticles = NapShiftParticles.size();
    NapShiftParticlesArray.initialize<int>(cu, numNapShiftParticles, "NapShiftParticlesArray");
    NapShiftParticlesArray.uploadSubArray(NapShiftParticles.data(), 0, numNapShiftParticles, true);

    angleForceArray.initialize<float>(cu, numPeptides*3*2*numInputAngles*4*3, "NapShiftAngleForceArray");
    NapShiftForceVector = torch::empty({numPeptides*3*2*numInputAngles*4, 3}, realOptionsDevice);
    inputTensor = torch::zeros({numPeptides, fullInputVecSize}, realOptionsCPU);

    std::vector<int> angleIndicesTemp(numPeptides*numInputAngles*4);
    std::vector<int> forceIndicesVector((numPeptides)*2*3*numInputAngles*4);
    //go through peptides and populate CSEsp, CSRC, CSScale, inputvec (BLOSUM), and record indices of particles associated with angles and NapShift forces
    for (int i=0; i<(numPeptides)*2*3*numInputAngles*4; i++)
        forceIndicesVector[i] = -1;

    std::vector<std::vector<int>> particleForceIndices = std::vector<std::vector<int>>(numNapShiftParticles);
    for (int i=0; i<numPeptides; i++) {
        if (validPeptide(i)) {
            for (int a=0; a<numAtomTypes; a++) {
                randomCoilTensor[i][a] = peptides[i]->csRC[atomTypes[a]];
                CSExpTensor[i][a] = peptides[i]->csExp[atomTypes[a]];
                ChemShiftScale[i][a] = peptides[i]->csScale[atomTypes[a]];
            }
            for (int j=0; j<22; j++) {
                if (neighbouringPeptides(i, i-1, 1)) inputTensor[i][j] = BLOSUM[peptides[i-1]->resType][j];
                inputTensor[i][oneInputVecSize+j] = BLOSUM[peptides[i]->resType][j];
                if (neighbouringPeptides(i, i+1, 1)) inputTensor[i][2*oneInputVecSize+j] = BLOSUM[peptides[i+1]->resType][j];
            }
        }

        std::vector<int*> inputAngles;

        if (modelFormat == 0 || modelFormat == 1) { //CG format
            int c = validPeptide(i) ? peptides[i]->getParticleIndices()[0] : -1;
            int sc = validPeptide(i) ? peptides[i]->getParticleIndices()[1] : -1;
            int l2 = neighbouringPeptides(i, i-2, 2) ?  peptides[i-2]->getParticleIndices()[0] : -1;
            int l1 = neighbouringPeptides(i, i-1, 1) ? peptides[i-1]->getParticleIndices()[0] : -1;
            int r1 = neighbouringPeptides(i, i+1, 1) ? peptides[i+1]->getParticleIndices()[0] : -1;
            int r2 = neighbouringPeptides(i, i+2, 2) ? peptides[i+2]->getParticleIndices()[0] : -1;

            int leftDihedral[] = {l2, l1, c, r1};
            int rightDihedral[] = {l1, c, r1, r2};
            int alpha[] = {l1, c, sc, -1};
            int beta[] = {r1, c, l1, -1};
            int gamma[] = {sc, c, r1, -1};

            if (modelFormat == 0) inputAngles = {leftDihedral, rightDihedral, alpha, beta, gamma}; //martini
            else inputAngles = {leftDihedral, rightDihedral, beta}; //CA
        } 
        else { //all-atom format
            
            int C_minus1 = neighbouringPeptides(i, i-1, 1) ? peptides[i-1]->getParticleIndices()[1] : -1;

            int N = validPeptide(i) ? peptides[i]->getParticleIndices()[0] : -1;
            int C = validPeptide(i) ? peptides[i]->getParticleIndices()[1] : -1;
            int CA = validPeptide(i) ? peptides[i]->getParticleIndices()[2] : -1;
            int CB = validPeptide(i) ? peptides[i]->getParticleIndices()[3] : -1;
            int G = validPeptide(i) ? peptides[i]->getParticleIndices()[4] : -1;
            int D = validPeptide(i) ? peptides[i]->getParticleIndices()[5] : -1;

            int N_plus1 = neighbouringPeptides(i, i+1, 1) ? peptides[i+1]->getParticleIndices()[0] : -1;
            

            int phi[] = {C_minus1, N, CA, C};
            int psi[] = {N, CA, C, N_plus1};
            int chi1[] = {N, CA, CB, G};
            int chi2[] = {CA, CB, G, D};

            inputAngles = {phi, psi, chi1, chi2};
        }

        //record indices of particles involved in angles and NapShift forces
        for (int angle=0; angle<numInputAngles; angle++){
            for (int particle=0; particle<4; particle++) {
                int system_particle = inputAngles[angle][particle];
                int particleIndex = find(NapShiftParticles.begin(), NapShiftParticles.end(), system_particle) - NapShiftParticles.begin();
                angleIndicesTemp[4*numInputAngles*i+angle*4+particle] = system_particle;

                if (system_particle > -1 && i > 0) particleForceIndices[particleIndex].push_back(3*2*4*numInputAngles*(i-1)+2*(2*4*numInputAngles)+4*(2*angle)+particle);
                if (system_particle > -1 && i > 0) particleForceIndices[particleIndex].push_back(3*2*4*numInputAngles*(i-1)+2*(2*4*numInputAngles)+4*(2*angle+1)+particle);
                if (system_particle > -1) particleForceIndices[particleIndex].push_back(3*2*4*numInputAngles*(i)+(2*4*numInputAngles)+4*(2*angle)+particle);
                if (system_particle > -1) particleForceIndices[particleIndex].push_back(3*2*4*numInputAngles*(i)+(2*4*numInputAngles)+4*(2*angle+1)+particle);
                if (system_particle > -1 && i < numPeptides-1) particleForceIndices[particleIndex].push_back(3*2*4*numInputAngles*(i+1)+4*(2*angle)+particle);
                if (system_particle > -1 && i < numPeptides-1) particleForceIndices[particleIndex].push_back(3*2*4*numInputAngles*(i+1)+4*(2*angle+1)+particle);
            }
        }
        
    }

    angleIndicesArray.initialize<int>(cu, numPeptides*numInputAngles*4, "NapShiftAngleIndicesArray");
    angleIndicesArray.uploadSubArray(angleIndicesTemp.data(), 0, numPeptides*numInputAngles*4, true);
        
    randomCoilTensor = randomCoilTensor.to(device);
    CSExpTensor = CSExpTensor.to(device);
    ChemShiftScale = ChemShiftScale.to(device);
    inputTensor = inputTensor.to(device);
    inputTensor.requires_grad_();

    for (int i=0; i<numNapShiftParticles; i++) {
        while (particleForceIndices[i].size() < fullInputVecSize) {
            particleForceIndices[i].push_back(-1);
        }
    }
    std::vector<int> linearParticleForceIndices = linearize(particleForceIndices);
    particleForceIndicesArray.initialize<int>(cu, numNapShiftParticles*fullInputVecSize, "NapShiftParticleForceIndicesArray");
    particleForceIndicesArray.uploadSubArray(linearParticleForceIndices.data(), 0, numNapShiftParticles*fullInputVecSize, true);

    // Pop the PyToch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that PyTorch haven't messed up the context stack

    // Initialize CUDA objects for OpenMM-NapShift
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = usePeriodic ? "1" : "0";
    ContextSelector selector(cu); // Switch to the OpenMM context
    CUmodule program = cu.createModule(CudaNapShiftKernelSources::napshiftForce, replacements);
    resolveAngleKernel = cu.getKernel(program, "resolveAngle");
    swapAtomToIndexKernel = cu.getKernel(program, "swapAtomToIndex");
    accumulateParticleForcesKernel = cu.getKernel(program, "accumulateParticleForces");
    DownloadCSDifferenceAvgDataKernel = cu.getKernel(program, "DownloadCSDifferenceAvgData");

    auto properties = force.getProperties();
    const std::string useCUDAGraphsString = properties["useCUDAGraphs"];
    if (useCUDAGraphsString == "true") {
        useGraphs = true;
    } else if (useCUDAGraphsString == "false" || useCUDAGraphsString == "") {
        useGraphs = false;
    } else {
        throw OpenMMException("NapShiftForce: invalid value of \"useCUDAGraphs\"");
    }
    if (useGraphs) {
        this->warmupSteps = std::stoi(properties["CUDAGraphWarmupSteps"]);
        if (this->warmupSteps <= 0) {
            throw OpenMMException("NapShiftForce: \"CUDAGraphWarmupSteps\" must be a positive integer");
        }
    }
    std::cout << "no NN execute" << std::endl;
}

void CudaCalcNapShiftForceKernel::prepareNapShiftInputs(ContextImpl& context) { 
    void* inputData = getTensorPointer(cu, inputTensor);

    int block_size = round64(round((numPeptides*numInputAngles-1)/(device_multiprocessors-1)));
    block_size = std::min(block_size, device_max_threads_per_block);

    CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context");
    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        void* inputArgs[] = {&inputData,
                             &angleForceArray.getDevicePointer(),
                             &angleIndicesArray.getDevicePointer(),
                             &indexToAtomArray.getDevicePointer(),
                             &cu.getPosq().getDevicePointer(),
                             &numPeptides,
                             &numInputAngles,
                             cu.getPeriodicBoxSizePointer(),
                             cu.getInvPeriodicBoxSizePointer() };
        
        cu.executeKernel(resolveAngleKernel, inputArgs, numPeptides*numInputAngles, block_size);
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
    }
}

void CudaCalcNapShiftForceKernel::accumulateParticleForces() { 
    void* NapShiftForceData = getTensorPointer(cu, NapShiftForceVector);

    int block_size = round64(round((numNapShiftParticles-1)/(device_multiprocessors-1)));
    block_size = std::min(block_size, device_max_threads_per_block);

    CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); 
    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* inputArgs[] = {&NapShiftForceData,
                             &angleForceArray.getDevicePointer(),
                             &particleForceIndicesArray.getDevicePointer(),
                             &NapShiftParticlesArray.getDevicePointer(),
                             &fullInputVecSize,
                             &numNapShiftParticles,
                             &indexToAtomArray.getDevicePointer(),
                             &cu.getForce().getDevicePointer(),
                             &paddedNumAtoms};
        cu.executeKernel(accumulateParticleForcesKernel, inputArgs, numNapShiftParticles, block_size);
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
    } 
}

void CudaCalcNapShiftForceKernel::getIndexToAtom() {
    int numParticles = cu.getNumAtoms();

    int block_size = round64(round((numParticles-1)/(device_multiprocessors-1)));
    block_size = std::min(block_size, device_max_threads_per_block);

    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        void* inputArgs[] = {&cu.getAtomIndexArray().getDevicePointer(),
                             &numParticles,
                             &indexToAtomArray.getDevicePointer()};
        cu.executeKernel(swapAtomToIndexKernel, inputArgs, numParticles, block_size);
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
    }
}

static void executeGraph(bool includeForces,
                         torch::jit::script::Module& predictionModel,
                         vector<torch::jit::IValue>& inputs,
                         torch::Tensor& inputTensor,
                         torch::Tensor& randomCoilTensor,
                         torch::Tensor& CSExpTensor,
                         torch::Tensor& NapShiftForceVector,
                         torch::Tensor& forcesToParticles,
                         torch::Tensor& modelErrors,
                         torch::Tensor& ChemShiftSTD,
                         torch::Tensor& ChemShiftScale,
                         int& numPeptides,
                         int& numNapShiftParticles,
                         int& numInputAngles,
                         torch::Tensor& K,
                         torch::Tensor& csTensor,
                         torch::Tensor& csDifference,
                         torch::Tensor& energyTensor,
                         torch::Tensor& dNN_dAngle,
                         torch::Tensor& relevant_dNN_dAngle) {
    NapShiftForceVector.zero_();
    //dNN_dAngle.zero_();

    csTensor = predictionModel.forward(inputs).toTensor();
    
    csDifference = CSExpTensor - (csTensor + randomCoilTensor);
    //dealing with undefined inputs
    csDifference = torch::where((CSExpTensor == -1) | (randomCoilTensor == -1), 0.0, csDifference); 
    //apply flatbottom
    csDifference = torch::where(torch::abs(csDifference) < modelErrors, 0.0, torch::abs(csDifference) - modelErrors); 
    energyTensor = K*torch::sum(ChemShiftScale*torch::square(csDifference)); 

    dNN_dAngle = torch::autograd::grad({energyTensor}, {inputTensor})[0].detach();
    //extract derivatives relating only to angles
    relevant_dNN_dAngle = torch::cat({torch::narrow(dNN_dAngle, 1, 22, 2*numInputAngles), torch::narrow(dNN_dAngle, 1, 44+2*numInputAngles, 2*numInputAngles), torch::narrow(dNN_dAngle, 1, 66+4*numInputAngles, 2*numInputAngles)}, 1).reshape({numPeptides, 3*2*numInputAngles, 1, 1}).repeat({1, 1, 4, 3});

    NapShiftForceVector += relevant_dNN_dAngle.view({-1, 3}).clone();
}  

static void executeGraphEnsembleAvg(bool includeForces,
                         torch::jit::script::Module& predictionModel,
                         vector<torch::jit::IValue>& inputs,
                         torch::Tensor& inputTensor,
                         torch::Tensor& randomCoilTensor,
                         torch::Tensor& CSExpTensor,
                         torch::Tensor& NapShiftForceVector,
                         torch::Tensor& forcesToParticles,
                         torch::Tensor& modelErrors,
                         torch::Tensor& ChemShiftSTD,
                         torch::Tensor& ChemShiftScale,
                         int& numPeptides,
                         int& numNapShiftParticles,
                         int& numInputAngles,
                         torch::Tensor& K,
                         torch::Tensor& csTensor,
                         torch::Tensor& avgCSTensor) {      

    csTensor = predictionModel.forward(inputs).toTensor();

    torch::Tensor csDifference = CSExpTensor - (csTensor + randomCoilTensor);
    csDifference = torch::where(CSExpTensor == -1, 0.0, csDifference);
    csDifference = torch::where(randomCoilTensor == -1, 0.0, csDifference);
    // --- flatbottom ---
    csDifference = torch::where(torch::abs(csDifference) < modelErrors, 0.0, csDifference);
    csDifference = torch::where(csDifference < 0, csDifference + modelErrors, csDifference);
    csDifference = torch::where(csDifference > 0, csDifference - modelErrors, csDifference);

    // ------------------
    //ensemble averaging 
    torch::Tensor csDifferenceAvg = CSExpTensor - (avgCSTensor + randomCoilTensor);
    csDifferenceAvg = torch::where(CSExpTensor == -1, 0.0, csDifferenceAvg);
    csDifferenceAvg = torch::where(randomCoilTensor == -1, 0.0, csDifferenceAvg);
    //apply flatbottom 
    csDifferenceAvg = torch::where(torch::abs(csDifferenceAvg) < modelErrors, 0.0, csDifferenceAvg);
    csDifferenceAvg = torch::where(csDifferenceAvg < 0, csDifferenceAvg + modelErrors, csDifferenceAvg); 
    csDifferenceAvg = torch::where(csDifferenceAvg > 0, csDifferenceAvg - modelErrors, csDifferenceAvg);
    // ------------------

    torch::Tensor dNN_dAngleN = torch::autograd::grad({torch::sum(csTensor.transpose(0,1)[0])}, {inputTensor}, {}, true)[0];
    torch::Tensor dNN_dAngleC = torch::autograd::grad({torch::sum(csTensor.transpose(0,1)[1])}, {inputTensor}, {}, true)[0];
    torch::Tensor dNN_dAngleCA = torch::autograd::grad({torch::sum(csTensor.transpose(0,1)[2])}, {inputTensor}, {}, true)[0];
    torch::Tensor dNN_dAngleCB = torch::autograd::grad({torch::sum(csTensor.transpose(0,1)[3])}, {inputTensor}, {}, true)[0];
    torch::Tensor dNN_dAngleH = torch::autograd::grad({torch::sum(csTensor.transpose(0,1)[4])}, {inputTensor}, {}, true)[0];
    torch::Tensor dNN_dAngleHA = torch::autograd::grad({torch::sum(csTensor.transpose(0,1)[5])}, {inputTensor}, {}, true)[0];

    torch::Tensor dNapShift_dAngle = -2*K*(ChemShiftScale.transpose(0,1)[0].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*csDifferenceAvg.transpose(0,1)[0].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*dNN_dAngleN +
                                           ChemShiftScale.transpose(0,1)[1].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*csDifferenceAvg.transpose(0,1)[1].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*dNN_dAngleC +
                                           ChemShiftScale.transpose(0,1)[2].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*csDifferenceAvg.transpose(0,1)[2].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*dNN_dAngleCA +
                                           ChemShiftScale.transpose(0,1)[3].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*csDifferenceAvg.transpose(0,1)[3].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*dNN_dAngleCB +
                                           ChemShiftScale.transpose(0,1)[4].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*csDifferenceAvg.transpose(0,1)[4].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*dNN_dAngleH +
                                           ChemShiftScale.transpose(0,1)[5].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*csDifferenceAvg.transpose(0,1)[5].unsqueeze(1).repeat({1,3*(22+2*numInputAngles)})*dNN_dAngleHA);                                    
    
    torch::Tensor relevant_dNapShift_dAngle = torch::cat({torch::narrow(dNapShift_dAngle, 1, 22, 2*numInputAngles), torch::narrow(dNapShift_dAngle, 1, 44+2*numInputAngles, 2*numInputAngles), torch::narrow(dNapShift_dAngle, 1, 66+4*numInputAngles, 2*numInputAngles)}, 1).reshape({numPeptides, 3*2*numInputAngles, 1, 1}).repeat({1, 1, 4, 3});                      

    torch::Tensor particleForceTensor = relevant_dNapShift_dAngle;
    NapShiftForceVector += particleForceTensor.view({-1, 3}).clone();
}  

double CudaCalcNapShiftForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
        long long currentStep = context.getStepCount();
    
    // Push to the PyTorch context
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");
    
    getIndexToAtom();

    prepareNapShiftInputs(context);

    K.zero_();
    K += context.getParameter("NapShift_K");
    if (ensembleAveraging) {
        DownloadCSDifferenceAvgs(context);
        UploadMyCSDifference(context);
    }

    vector<torch::jit::IValue> inputs;
    inputs = {inputTensor};

    if (!useGraphs) { 
            if (!ensembleAveraging || currentStep == 0) {
                executeGraph(includeForces,
                    model,
                    inputs,
                    inputTensor,
                    randomCoilTensor,
                    CSExpTensor,
                    NapShiftForceVector,
                    forcesToParticles,
                    modelErrors,
                    ChemShiftSTD,
                    ChemShiftScale,
                    numPeptides,
                    numNapShiftParticles,
                    numInputAngles,
                    K,
                    csTensor,
                    csDifference,
                    energyTensor,
                    dNN_dAngle,
                    relevant_dNN_dAngle);
            } else {
                executeGraphEnsembleAvg(includeForces,
                    model,
                    inputs,
                    inputTensor,
                    randomCoilTensor,
                    CSExpTensor,
                    NapShiftForceVector,
                    forcesToParticles,
                    modelErrors,
                    ChemShiftSTD,
                    ChemShiftScale,
                    numPeptides,
                    numNapShiftParticles,
                    numInputAngles,
                    K,
                    csTensor,
                    avgCSTensor);
            }
    } else {
        // Record graph if not already done
        bool is_graph_captured = false;
        if (graphs.find(includeForces) == graphs.end()) {
            //CUDA graph capture must occur in a non-default stream
            const auto stream = c10::cuda::getStreamFromPool(false, cu.getDeviceIndex());
            const c10::cuda::CUDAStreamGuard guard(stream);
            // Warmup the graph workload before capturing.  This first
            // run  before  capture sets  up  allocations  so that  no
            // allocations are  needed after.  Pytorch's  allocator is
            // stream  capture-aware and,  after warmup,  will provide
            // record static pointers and shapes during capture.
            try {
                for (int i = 0; i < this->warmupSteps; i++)
                    if (!ensembleAveraging || currentStep == 0) {
                        executeGraph(includeForces,
                            model,
                            inputs,
                            inputTensor,
                            randomCoilTensor,
                            CSExpTensor,
                            NapShiftForceVector,
                            forcesToParticles,
                            modelErrors,
                            ChemShiftSTD,
                            ChemShiftScale,
                            numPeptides,
                            numNapShiftParticles,
                            numInputAngles,
                            K,
                            csTensor,
                            csDifference,
                            energyTensor,
                            dNN_dAngle,
                            relevant_dNN_dAngle);                        
                    } else {
                        executeGraphEnsembleAvg(includeForces,
                            model,
                            inputs,
                            inputTensor,
                            randomCoilTensor,
                            CSExpTensor,
                            NapShiftForceVector,
                            forcesToParticles,
                            modelErrors,
                            ChemShiftSTD,
                            ChemShiftScale,
                            numPeptides,
                            numNapShiftParticles,
                            numInputAngles,
                            K,
                            csTensor,
                            avgCSTensor);  
                    }
            }
            catch (std::exception& e) {
                throw OpenMMException(string("NapShiftForce: Failed to warmup the model before graph construction. PyTorch reported the following error:\n") + e.what());
            }
            graphs[includeForces].capture_begin();
            try {
                if (!ensembleAveraging || currentStep == 0) {
                    executeGraph(includeForces,
                        model,
                        inputs,
                        inputTensor,
                        randomCoilTensor,
                        CSExpTensor,
                        NapShiftForceVector,
                        forcesToParticles,
                        modelErrors,
                        ChemShiftSTD,
                        ChemShiftScale,
                        numPeptides,
                        numNapShiftParticles,
                        numInputAngles,
                        K,
                        csTensor,
                        csDifference,
                        energyTensor,
                        dNN_dAngle,
                        relevant_dNN_dAngle);
                } else {
                    executeGraphEnsembleAvg(includeForces,
                        model,
                        inputs,
                        inputTensor,
                        randomCoilTensor,
                        CSExpTensor,
                        NapShiftForceVector,
                        forcesToParticles,
                        modelErrors,
                        ChemShiftSTD,
                        ChemShiftScale,
                        numPeptides,
                        numNapShiftParticles,
                        numInputAngles,
                        K,
                        csTensor,
                        avgCSTensor);
                }
                is_graph_captured = true;
                graphs[includeForces].capture_end();
            }
            catch (std::exception& e) {
                if (!is_graph_captured) {
                    graphs[includeForces].capture_end();
                }
                throw OpenMMException(string("NapShiftForce: Failed to capture the model into a CUDA graph. PyTorch reported the following error:\n") + e.what());
            }
        }
        // Use the same stream as the OpenMM context, even if it is the default stream
        const auto openmmStream = cu.getCurrentStream();
        const auto stream = c10::cuda::getStreamFromExternal(openmmStream, cu.getDeviceIndex());
        const c10::cuda::CUDAStreamGuard guard(stream);

        graphs[includeForces].replay();   
    }

    if (includeForces) {
        if (context.getParameter("NapShift_K") > 0){
            accumulateParticleForces();
        }
    }
    
    // Pop to the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that the correct context was popped

    return 0.0;
}

bool CudaCalcNapShiftForceKernel::validPeptide(int index){
    return index >= 0 && index < numPeptides;
}

bool CudaCalcNapShiftForceKernel::neighbouringPeptides(int peptideIdx1, int peptideIdx2, int distance){
    if ( !validPeptide(peptideIdx1) || !validPeptide(peptideIdx2) ) return false;
    if ( peptides[peptideIdx1]->getChainId() != peptides[peptideIdx2]->getChainId() ) return false;
    if ( (peptides[peptideIdx1]->getResId() - peptides[peptideIdx2]->getResId() != distance) && ( peptides[peptideIdx2]->getResId() -  peptides[peptideIdx1]->getResId() != distance)) return false;
    return true;
}

int CudaCalcNapShiftForceKernel::NapShiftIndex(int systemIndex){
    if (systemIndex < 0) return numNapShiftParticles;
    return find(NapShiftParticles.begin(), NapShiftParticles.end(), systemIndex) - NapShiftParticles.begin();
}

void CudaCalcNapShiftForceKernel::DownloadCSDifferenceAvgs(ContextImpl& context) { 
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex());
    avgCSTensor.zero_();
    std::vector<float> temp(numAtomTypes*numPeptides);
    for (int i=0; i<numPeptides; i++){
        for (int a=0; a<numAtomTypes; a++) {
            temp[i*numAtomTypes+a] = context.getParameter("NapShift_AvgCSDifference" + std::to_string(i) + atomTypes[a]);
        }
    }
    torch::Tensor tempTensor = torch::from_blob(temp.data(), {numPeptides, numAtomTypes}, realOptionsCPU);
    tempTensor = tempTensor.to(device);

    void* avgCSData = getTensorPointer(cu, avgCSTensor);
    void* tempData = getTensorPointer(cu, tempTensor);

    int block_size = round64(round((numNapShiftParticles-1)/(device_multiprocessors-1)));
    block_size = std::min(block_size, device_max_threads_per_block);

    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        void* inputArgs[] = {&avgCSData,
                             &tempData,
                             &numNapShiftParticles};
        cu.executeKernel(DownloadCSDifferenceAvgDataKernel, inputArgs, numNapShiftParticles, block_size);
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
    }
}

void CudaCalcNapShiftForceKernel::UploadMyCSDifference(ContextImpl& context) { 
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex());
    torch::Tensor tempTensor = csTensor.clone();
    tempTensor = tempTensor.to("cpu");
    auto tempTensor_a = tempTensor.accessor<float,2>();
    for (int i=0; i<numPeptides; i++){
        for (int a=0; a<numAtomTypes; a++) {
            context.setParameter("NapShift_MyCSDifference" + std::to_string(i) + atomTypes[a], tempTensor_a[i][a]);
        }
    }
}

