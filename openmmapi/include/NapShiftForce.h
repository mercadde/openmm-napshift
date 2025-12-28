#ifndef OPENMM_NAPSHIFTFORCE_H_
#define OPENMM_NAPSHIFTFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <torch/torch.h>
#include "internal/windowsExportNapShift.h"

namespace NapShiftPlugin {

/**
 * This class implements a restraining force based on user-supplied experimental Chemical Shifts.*/

class OPENMM_EXPORT_NAPSHIFT NapShiftForce : public OpenMM::Force {
public:
    NapShiftForce(const std::map<std::string, std::string>& properties = {});
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;  
    void setUsesEnsembleAveraging(bool ensembleAveraging);  
    bool usesEnsembleAveraging() const;    
    int addPeptide(int bbIndex, int scIndex, char resType, std::map<std::string, double> csExp1, std::map<std::string, double> csExp2, std::map<std::string, double> csRC, std::map<std::string, double> csScale, int resId, std::string chainId);
    int addPeptide(int N, int C, int CA, int CB, int G, int D, char resType, std::map<std::string, double> csExp1, std::map<std::string, double> csExp2, std::map<std::string, double> csRC, std::map<std::string, double> csScale, int resId, std::string chainId);
    void getPeptideParameters(int index, int& bbIndex, int& scIndex, char& resType, std::map<std::string, double>& csExp1, std::map<std::string, double>& csExp2, std::map<std::string, double>& csRC, std::map<std::string, double>& csScale, int& resId, std::string& chainId) const;
    void getPeptideParameters(int index, int& N, int& C, int& CA, int& CB, int& G, int& D, char& resType, std::map<std::string, double>& csExp1, std::map<std::string, double>& csExp2, std::map<std::string, double>& csRC, std::map<std::string, double>& csScale, int& resId, std::string& chainId) const;
    void setPeptideParameters(int index, int& bbIndex, int& scIndex, char& resType, std::map<std::string, double>& csExp1, std::map<std::string, double>& csExp2, std::map<std::string, double>& csRC, std::map<std::string, double>& csScale, int& resId, std::string& chainId);
    void setPeptideParameters(int index, int& N, int& C, int& CA, int& CB, int& G, int& D, char& resType, std::map<std::string, double>& csExp1, std::map<std::string, double>& csExp2, std::map<std::string, double>& csRC, std::map<std::string, double>& csScale, int& resId, std::string& chainId);
    int getNumPeptides() const;
    std::vector<std::string> getAtoms() const;
    void setReportingParams(std::string reportFilename, int reportInterval);
    std::string getReportFilename() const;
    int getReportInterval() const;
    void setProperty(const std::string& name, const std::string& value);
    const std::map<std::string, std::string>& getProperties() const;
    /**
     * Get the type of ANN model to use: MARTINI3, CA-only, or All-Atom.
     */
    std::string getModelType() const;
    void setModelType(std::string modelType);
    /**
     * Get the default value of K, the CS-restraint force constant.
     */
    double getK1DefaultValue() const;
    void setK1DefaultValue(double defaultK1);
    double getK2DefaultValue() const;
    void setK2DefaultValue(double defaultK2);
    double getSigma1DefaultValue() const;
    void setSigma1DefaultValue(double defaultSigma1);
    double getSigma2DefaultValue() const;
    void setSigma2DefaultValue(double defaultSigma2);
    /**
     * Get the directory to source the ANN model from.
     */
    std::string getPytorchModelsDir() const;
    void setPytorchModelsDir(std::string pytorchModelsDir);


protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class BasePeptide;
    class Peptide;
    class AllAtomPeptide;


    std::map<std::string, std::string> properties;
    std::string emptyProperty;

    std::string modelType;

    bool usePeriodic;
    bool useEnsembleAveraging;
    bool uploadEnergyTensor;

    std::vector<std::unique_ptr<BasePeptide>> peptides; //TODO: serialization..?

    double K1, K2, sigma1, sigma2;
    std::vector<double> avgCS;
    std::vector<double> thisCS;

    double flatbottom_scale;

    std::string reportFilename;
    int reportInterval;

    std::string pytorchModelsDir;
};

class NapShiftForce::BasePeptide {
    public:
        char resType;
        std::map<std::string, double> csExp1;
        std::map<std::string, double> csExp2;
        std::map<std::string, double> csRC;
        std::map<std::string, double> csScale;
        int resId;
        std::string chainId;
        BasePeptide() {
            resType = '-';
            csExp1 = std::map<std::string, double> {{"N", -1}, {"C", -1}, {"CA", -1}, {"CB", -1}, {"H", -1}, {"HA", -1}};
            csExp2 = std::map<std::string, double> {{"N", -1}, {"C", -1}, {"CA", -1}, {"CB", -1}, {"H", -1}, {"HA", -1}};
            csRC = std::map<std::string, double> {{"N", -1}, {"C", -1}, {"CA", -1}, {"CB", -1}, {"H", -1}, {"HA", -1}};
            csScale = std::map<std::string, double> {{"N", 1}, {"C", 1}, {"CA", 1}, {"CB", 1}, {"H", 1}, {"HA", 1}};
            resId = -1;
            chainId = "";
        }
        BasePeptide(char resType, std::map<std::string, double> csExp1, std::map<std::string, double> csExp2, std::map<std::string, double> csRC, std::map<std::string, double> csScale, int resId, std::string chainId) {
            this->resType = resType;
            this->csExp1 = csExp2;
            this->csExp1 = csExp2;
            this->csRC = csRC;
            this->csScale = csScale;
            this->resId = resId;
            this->chainId = chainId;
        }
        virtual ~BasePeptide() = default;
        virtual std::vector<int> getParticleIndices() { 
            std::vector<int> indices = {};
            return indices;
        }
        virtual void setParticleIndices(int bbIndex, int scIndex) { }
        virtual void setParticleIndices(int N, int C, int CA, int CB, int G, int D) { }
};

class NapShiftForce::Peptide : public NapShiftForce::BasePeptide {
    public:
        int bbIndex, scIndex;
        Peptide() : BasePeptide() {
            bbIndex = scIndex = -1;
        }
        Peptide(int bbIndex, int scIndex, char resType, std::map<std::string, double> csExp1, std::map<std::string, double> csExp2, std::map<std::string, double> csRC, std::map<std::string, double> csScale, int resId, std::string chainId) :
            BasePeptide(resType, csExp1, csExp2, csRC, csScale, resId, chainId) {
                this->bbIndex = bbIndex;
                this->scIndex = scIndex;
        }
        std::vector<int> getParticleIndices() { 
            std::vector<int> indices = {bbIndex, scIndex};
            return indices;
        }
        void setParticleIndices(int bbIndex, int scIndex) { 
            this->bbIndex = bbIndex;
            this->scIndex = scIndex;
        }
};

class NapShiftForce::AllAtomPeptide : public NapShiftForce::BasePeptide {
    public:
        int N, C, CA, CB, G, D;
        AllAtomPeptide() : BasePeptide() {
            N = C = CA = CB = G = D -1;
        }
        AllAtomPeptide(int N, int C, int CA, int CB, int G, int D, char resType, std::map<std::string, double> csExp1, std::map<std::string, double> csExp2, std::map<std::string, double> csRC, std::map<std::string, double> csScale, int resId, std::string chainId) :
            BasePeptide(resType, csExp1, csExp2, csRC, csScale, resId, chainId) {
                this->N = N;
                this->C = C;
                this->CA = CA;
                this->CB = CB;
                this->G = G;
                this->D = D;
        }
        std::vector<int> getParticleIndices() { 
            std::vector<int> indices = {N, C, CA, CB, G, D};
            return indices;
        }
        void setParticleIndices(int N, int C, int CA, int CB, int G, int D) { 
            this->N = N;
            this->C = C;
            this->CA = CA;
            this->CB = CB;
            this->G = G;
            this->D = D;
        }
};

} // namespace NapShiftPlugin


#endif /*OPENMM_NAPSHIFTFORCE_H_*/
