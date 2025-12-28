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

#include "NapShiftForce.h"
#include "internal/NapShiftForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>
#include <filesystem>
#include <string>
#include <memory>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>

using namespace NapShiftPlugin;
using namespace OpenMM;
using namespace std;

static  std::string GetExecutableDir() {
        std::string subject = std::filesystem::canonical("/proc/self/exe").string();
        std::string search = "/bin/";
        std::string replace = "/lib/";
        size_t pos = 0;
        bool already_found_bin = false;
        while ((pos = subject.find(search, pos)) != std::string::npos) {
            if (already_found_bin) {
                throw OpenMMException("NapShiftForce: couldn't determine executable path!");
            }
            subject.replace(pos, search.length(), replace);
            pos += replace.length();
            already_found_bin = true;
        }
        return subject;
}

NapShiftForce::NapShiftForce(const map<string, string>& properties) : usePeriodic(false) {
    const std::map<std::string, std::string> defaultProperties = {{"useCUDAGraphs", "true"}, {"CUDAGraphWarmupSteps", "100"}};
    this->properties = defaultProperties;
    for (auto& property : properties) {
        if (defaultProperties.find(property.first) == defaultProperties.end())
            throw OpenMMException("NapShiftForce: Unknown property '" + property.first + "'");
        this->properties[property.first] = property.second;
    }
    modelType = "martini";
    K1 = 0.0;
    K2 = 0.0;
    sigma1 = 1.0;
    sigma2 = 1.0;
    avgCS = {};
    useEnsembleAveraging = false;
    pytorchModelsDir = GetExecutableDir() + "/site-packages/openmmnapshift/PytorchModels";
}

ForceImpl* NapShiftForce::createImpl() const {
    return new NapShiftForceImpl(*this);
}
void NapShiftForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}
bool NapShiftForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}
void NapShiftForce::setUsesEnsembleAveraging(bool ensembleAveraging) {
    useEnsembleAveraging = ensembleAveraging;
}
bool NapShiftForce::usesEnsembleAveraging() const {
    return useEnsembleAveraging;
}

int NapShiftForce::addPeptide(int bbIndex, int scIndex, char resType, std::map<std::string, double> csExp1, std::map<std::string, double> csExp2,  std::map<std::string, double> csRC, std::map<std::string, double> csScale, int resId, std::string chainId){
    peptides.push_back(std::unique_ptr<Peptide>(std::make_unique<Peptide>(bbIndex, scIndex, resType, csExp1, csExp2, csRC, csScale, resId, chainId)));
    return peptides.size()-1;
}
int NapShiftForce::addPeptide(int N, int C, int CA, int CB, int G, int D, char resType, std::map<std::string, double> csExp1, std::map<std::string, double> csExp2, std::map<std::string, double> csRC, std::map<std::string, double> csScale, int resId, std::string chainId){
    peptides.push_back(std::unique_ptr<AllAtomPeptide>(std::make_unique<AllAtomPeptide>(N, C, CA, CB, G, D, resType, csExp1, csExp2, csRC, csScale, resId, chainId)));
    return peptides.size()-1;
}

void NapShiftForce::getPeptideParameters(int index, int& bbIndex, int& scIndex, char& resType, std::map<std::string, double>& csExp1, std::map<std::string, double>& csExp2, std::map<std::string, double>& csRC, std::map<std::string, double>& csScale, int& resId, std::string& chainId) const {
    ASSERT_VALID_INDEX(index, peptides);
    bbIndex = peptides[index]->getParticleIndices()[0];
    scIndex = peptides[index]->getParticleIndices()[1];
    resType = peptides[index]->resType;
    csExp1 = peptides[index]->csExp1;
    csExp2 = peptides[index]->csExp2;
    csRC = peptides[index]->csRC;
    csScale = peptides[index]->csScale;
    resId = peptides[index]->resId;
    chainId = peptides[index]->chainId;
}

void NapShiftForce::getPeptideParameters(int index, int& N, int& C, int& CA, int& CB, int& G, int& D, char& resType, std::map<std::string, double>& csExp1, std::map<std::string, double>& csExp2, std::map<std::string, double>& csRC, std::map<std::string, double>& csScale, int& resId, std::string& chainId) const {
    ASSERT_VALID_INDEX(index, peptides);
    N = peptides[index]->getParticleIndices()[0];
    C = peptides[index]->getParticleIndices()[1];
    CA = peptides[index]->getParticleIndices()[2];
    CB = peptides[index]->getParticleIndices()[3];
    G = peptides[index]->getParticleIndices()[4];
    D = peptides[index]->getParticleIndices()[5];
    resType = peptides[index]->resType;
    csExp1 = peptides[index]->csExp1;
    csExp2 = peptides[index]->csExp2;
    csRC = peptides[index]->csRC;
    csScale = peptides[index]->csScale;
    resId = peptides[index]->resId;
    chainId = peptides[index]->chainId;
}

void NapShiftForce::setPeptideParameters(int index, int& bbIndex, int& scIndex, char& resType, std::map<std::string, double>& csExp1, std::map<std::string, double>& csExp2, std::map<std::string, double>& csRC, std::map<std::string, double>& csScale, int& resId, std::string& chainId) {
    ASSERT_VALID_INDEX(index, peptides);
    peptides[index]->setParticleIndices(bbIndex, scIndex);
    peptides[index]->resType = resType;
    peptides[index]->csExp1 = csExp1;
    peptides[index]->csExp2 = csExp2;
    peptides[index]->csScale = csScale;
    peptides[index]->csRC = csRC;
    peptides[index]->resId = resId;
    peptides[index]->chainId = chainId;
}

void NapShiftForce::setPeptideParameters(int index, int& N, int& C, int& CA, int& CB, int& G, int& D, char& resType, std::map<std::string, double>& csExp1, std::map<std::string, double>& csExp2, std::map<std::string, double>& csRC, std::map<std::string, double>& csScale, int& resId, std::string& chainId) {
    ASSERT_VALID_INDEX(index, peptides);
    peptides[index]->setParticleIndices(N, C, CA, CB, G, D);
    peptides[index]->resType = resType;
    peptides[index]->csExp1 = csExp1;
    peptides[index]->csExp2 = csExp2;
    peptides[index]->csScale = csScale;
    peptides[index]->csRC = csRC;
    peptides[index]->resId = resId;
    peptides[index]->chainId = chainId;
}

int NapShiftForce::getNumPeptides() const {
    return peptides.size();
}

std::vector<std::string> NapShiftForce::getAtoms() const {
    return {"N", "C", "CA", "CB", "H", "HA"};
}

void NapShiftForce::setProperty(const std::string& name, const std::string& value) {
    if (properties.find(name) == properties.end())
        throw OpenMMException("NapShiftForce: Unknown property '" + name + "'");
    properties[name] = value;
}

const std::map<std::string, std::string>& NapShiftForce::getProperties() const {
    return properties;
}

void NapShiftForce::setModelType(std::string modelType) {
    this->modelType = modelType;
}

std::string NapShiftForce::getModelType() const {
    return modelType;
}

double NapShiftForce::getK1DefaultValue() const {
    return K1;
}

void NapShiftForce::setK1DefaultValue(double defaultK1) {
    K1 = defaultK1;
}

double NapShiftForce::getK2DefaultValue() const {
    return K2;
}

void NapShiftForce::setK2DefaultValue(double defaultK2) {
    K2 = defaultK2;
}

double NapShiftForce::getSigma1DefaultValue() const {
    return sigma1;
}

void NapShiftForce::setSigma1DefaultValue(double defaultSigma1) {
    sigma1 = defaultSigma1;
}

double NapShiftForce::getSigma2DefaultValue() const {
    return sigma2;
}

void NapShiftForce::setSigma2DefaultValue(double defaultSigma2) {
    sigma2 = defaultSigma2;
}

std::string NapShiftForce::getPytorchModelsDir() const {
    return pytorchModelsDir;
}

void NapShiftForce::setPytorchModelsDir(std::string pytorchModelsDir) {
    this->pytorchModelsDir = pytorchModelsDir;
}