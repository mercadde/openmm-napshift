#ifndef CUDA_NAPSHIFT_KERNELS_H_
#define CUDA_NAPSHIFT_KERNELS_H_

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

#include "NapShiftKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <torch/version.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/ATen.h>
#include <set>
#include <memory>

namespace NapShiftPlugin {

/**
 * This kernel is invoked by NapShiftForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcNapShiftForceKernel : public CalcNapShiftForceKernel {
public:
    CudaCalcNapShiftForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu);
    ~CudaCalcNapShiftForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system         the System this kernel will be applied to
     * @param force          the NapShiftForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const NapShiftForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return energy of 0.0 since this is a restraint potential
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

private:

    bool hasInitializedKernel;
    OpenMM::CudaContext& cu;
    torch::TensorOptions realOptionsDevice;
    torch::TensorOptions intOptionsDevice;
    torch::TensorOptions realOptionsCPU;
    torch::TensorOptions intOptionsCPU;

    bool usePeriodic;
    bool ensembleAveraging;

    int device_multiprocessors;
    int device_max_threads_per_block;
    
    int modelFormat; //0=martini, 1=CA, 2=all_atom
    int numInputAngles;
    
    OpenMM::CudaArray indexToAtomArray; //maps system indices to renumbered GPU indices

    torch::Tensor inputTensor;
    int oneInputVecSize;
    int fullInputVecSize;

    int numNapShiftParticles;
    std::vector<int> NapShiftParticles;
    OpenMM::CudaArray NapShiftParticlesArray; //maps NapShift particle indices to system indices
    OpenMM::CudaArray angleIndicesArray; //(system) indices of atoms which make up each angle in each peptide
    OpenMM::CudaArray angleForceArray; //for each angle, stores forces on each constituent particle
    OpenMM::CudaArray particleForceIndicesArray; //for each particle, stores the indices of forces in particleForceVector which should be summed to give the total force on that particle

    torch::jit::script::Module model;
    at::Tensor K;

    torch::Tensor modelErrors;
    torch::Tensor ChemShiftSTD;
    torch::Tensor ChemShiftScale;
    torch::Tensor RMSD; 
    torch::Tensor randomCoilTensor;
    torch::Tensor CSExpTensor1;
    torch::Tensor CSExpTensor2;
    torch::Tensor csTensor;
    torch::Tensor NapShiftForceVector; //vector of all forces resulting from NapShift prediction

    //ensemble averaging
    torch::Tensor avgCSTensor;
    torch::Tensor dNapShift_dDelta;

    // ---
    torch::Tensor csDifference;
    torch::Tensor energyTensor;
    torch::Tensor dNN_dAngle;
    torch::Tensor relevant_dNN_dAngle;
    // ---

    torch::Tensor forcesToParticles;

    CUfunction accumulateParticleForcesKernel, resolveAngleKernel, swapAtomToIndexKernel, DownloadCSDifferenceAvgDataKernel;
    CUcontext primaryContext;
    
    std::map<bool, at::cuda::CUDAGraph> graphs;
    bool useGraphs;
    int warmupSteps;

    void prepareNapShiftInputs(OpenMM::ContextImpl& context);
    void accumulateParticleForces();
    void addForces();
    bool validPeptide(int index);
    bool neighbouringPeptides(int peptideIdx1, int peptideIdx2, int distance);
    int NapShiftIndex(int systemIndex);
    void getIndexToAtom();

    void DownloadCSDifferenceAvgs(OpenMM::ContextImpl& context);
    void UploadMyCSDifference(OpenMM::ContextImpl& context);

    void UploadEnergyTensor(OpenMM::ContextImpl& context);

    std::string reportFilename;
    int reportInterval;

    class Peptide;
    class AllAtomPeptide;
    class BasePeptide;
    int numPeptides;
    std::vector<std::unique_ptr<BasePeptide>> peptides;

    std::vector<std::string> atomTypes = {"N", "C", "CA", "CB", "H", "HA"};
    int numAtomTypes;

    std::map<std::string, float> martiniModelErrorMap = {{ "N", 2.566 },
                                                { "C", 1.187 }, 
                                                { "CA", 1.286 },
                                                { "CB", 1.314 },
                                                { "H", 0.425 },  
                                                { "HA", 0.284 }  
                                                };  
    std::map<std::string, float> CAModelErrorMap = {{ "N", 2.755 },
                                                { "C", 1.304 }, 
                                                { "CA", 1.327 },
                                                { "CB", 1.494 },
                                                { "H", 0.465 },  
                                                { "HA", 0.308 } 
                                                };  
    std::map<std::string, float> allAtomModelErrorMap = {{ "N", 2.706 },
                                                { "C", 1.265 }, 
                                                { "CA", 1.353 },
                                                { "CB", 1.416 },
                                                { "H", 0.459 },  
                                                { "HA", 0.301 } 
                                                };  

    std::map<char, std::vector<int>> BLOSUM = {
        {'A', {+4, -1, -2, -2, +0, +0, -1, -1, +0, -2, -1, -1, -1, -1, -2, -1, -1, +1, +0, -3, -2, +0} },
        {'R', {-1, +5, +0, -2, -3, -3, +1, +0, -2, +0, -3, -2, +2, -1, -3, -2, -2, -1, -1, -3, -2, -3} },
        {'N', {-2, +0, +6, +1, -3, -3, +0, +0, +0, +1, -3, -3, +0, -2, -3, -2, -2, +1, +0, -4, -2, -3} },
        {'D', {-2, -2, +1, +6, -3, -3, +0, +2, -1, -1, -3, -4, -1, -3, -3, -1, -1, +0, -1, -4, -3, -3} },
        {'X', {+0, -3, -3, -3, +9, +8, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -3, -1, -1, -2, -2, -1} }, //CYO
        {'C', {+0, -3, -3, -3, +8, +9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -3, -1, -1, -2, -2, -1} }, //CYR
        {'Q', {-1, +1, +0, +0, -3, -3, +5, +2, -2, +0, -3, -2, +1, +0, -3, -1, -1, +0, -1, -2, -1, -2} },
        {'E', {-1, +0, +0, +2, -4, -4, +2, +5, -2, +0, -3, -3, +1, -2, -3, -1, -1, +0, -1, -3, -2, -2} },
        {'G', {+0, -2, +0, -1, -3, -3, -2, -2, +6, -2, -4, -4, -2, -3, -3, -2, -2, +0, -2, -2, -3, -3} },
        {'H', {-2, +0, +1, -1, -3, -3, +0, +0, -2, +8, -3, -3, -1, -2, -1, -2, -2, -1, -2, -2, +2, -3} },
        {'I', {-1, -3, -3, -3, -1, -1, -3, -3, -4, -3, +4, +2, -3, +1, +0, -3, -3, -2, -1, -3, -1, +3} },
        {'L', {-1, -2, -3, -4, -1, -1, -2, -3, -4, -3, +2, +4, -2, +2, +0, -3, -3, -2, -1, -2, -1, +1} },
        {'K', {-1, +2, +0, -1, -3, -3, +1, +1, -2, -1, -3, -2, +5, -1, -3, -1, -1, +0, -1, -3, -2, -2} },
        {'M', {-1, -1, -2, -3, -1, -1, +0, -2, -3, -2, +1, +2, -1, +5, +0, -2, -2, -1, -1, -1, -1, +1} },
        {'F', {-2, -3, -3, -3, -2, -2, -3, -3, -3, -1, +0, +0, -3, +0, +6, -4, -4, -2, -2, +1, +3, -1} },
        {'O', {-1, -2, -2, -1, -3, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, +7, +6, -1, -1, -4, -3, -2} }, //PRC
        {'P', {-1, -2, -2, -1, -3, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, +6, +7, -1, -1, -4, -3, -2} }, //PRT
        {'S', {+1, -1, +1, +0, -1, -1, +0, +0, +0, -1, -2, -2, +0, -1, -2, -1, -1, +4, +1, -3, -2, -2} },
        {'T', {+0, -1, +0, -1, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, -1, +1, +5, -2, -2, +0} },
        {'W', {-3, -3, -4, -4, -2, -2, -2, -3, -2, -2, -3, -2, -3, -1, +1, -4, -4, -3, -2, 11, +2, -3} },
        {'Y', {-2, -2, -2, -3, -2, -2, -1, -2, -3, +2, -1, -1, -2, -1, +3, -3, -3, -2, -2, +2, +7, -1} },
        {'V', {+0, -3, -3, -3, -1, -1, -2, -2, -3, -3, +3, +1, -2, +1, -1, -2, -2, -2, +0, -3, -1, +4} },
        {'-', {+0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0} }
    };
    
};

class CudaCalcNapShiftForceKernel::BasePeptide {
    public:
        char resType;
        std::map<std::string, double> csExp1;
        std::map<std::string, double> csExp2;
        std::map<std::string, double> csRC;
        std::map<std::string, double> csScale;
        int resId;
        std::string chainId;
        BasePeptide(char resType,
                std::map<std::string,double> csExp1,
                std::map<std::string,double> csExp2,
                std::map<std::string, double> csRC,
                std::map<std::string, double> csScale,
                int resId, 
                std::string chainId) {
            this->resType = resType;
            this->csExp1 = csExp1;
            this->csExp2 = csExp2;
            this->csRC = csRC;
            this->csScale = csScale;
            this->resId = resId;
            this->chainId = chainId;
        }
        virtual ~BasePeptide() = default;
        int getResId() {   return resId;    }
        std::string getChainId() {   return chainId;    }
        virtual std::vector<int> getParticleIndices() { 
            std::vector<int> indices = {};
            return indices;
        }
};

class CudaCalcNapShiftForceKernel::Peptide : public CudaCalcNapShiftForceKernel::BasePeptide {
    public:
        int bbIndex;
        int scIndex;
        Peptide(int bbIndex,
                int scIndex,
                char resType,
                std::map<std::string, double> csExp1,
                std::map<std::string, double> csExp2,
                std::map<std::string, double> csRC,
                std::map<std::string, double> csScale,
                int resId, 
                std::string chainId) : BasePeptide(resType, csExp1, csExp2, csRC, csScale, resId, chainId) {
            this->bbIndex = bbIndex;
            this->scIndex = scIndex;
        }
        std::vector<int> getParticleIndices() { 
            std::vector<int> indices = {bbIndex, scIndex};
            return indices;
        }
};

class CudaCalcNapShiftForceKernel::AllAtomPeptide : public CudaCalcNapShiftForceKernel::BasePeptide {
    public:
        int N_index;
        int C_index;
        int CA_index;
        int CB_index;
        int G_index;
        int D_index;
        AllAtomPeptide(int N_index,
                int C_index,
                int CA_index,
                int CB_index,
                int G_index,
                int D_index,
                char resType,
                std::map<std::string, double> csExp1,
                std::map<std::string, double> csExp2,
                std::map<std::string, double> csRC,
                std::map<std::string, double> csScale,
                int resId, 
                std::string chainId) : BasePeptide(resType, csExp1, csExp2, csRC, csScale, resId, chainId) {
            this->N_index = N_index;
            this->C_index = C_index;
            this->CA_index = CA_index;
            this->CB_index = CB_index;
            this->G_index = G_index;
            this->D_index = D_index;
        }
        std::vector<int> getParticleIndices() { 
            std::vector<int> indices = {N_index, C_index, CA_index, CB_index, G_index, D_index};
            return indices;
        }
};

} // namespace NapShiftPlugin

#endif /*CUDA_NAPSHIFT_KERNELS_H_*/
