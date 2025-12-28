/* -------------------------------------------------------------------------- *
 *                                 OpenMM-NapShift                            *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2024 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
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

#include "NapShiftForceProxy.h"
#include "NapShiftForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/csrc/jit/serialization/import.h>

using namespace NapShiftPlugin;
using namespace OpenMM;
using namespace std;

static string hexEncode(const string& input) {
    stringstream ss;
    ss << hex << setfill('0');
    for (const unsigned char& i : input) {
        ss << setw(2) << static_cast<uint64_t>(i);
    }
    return ss.str();
}

static string hexDecode(const string& input) {
    string res;
    res.reserve(input.size() / 2);
    for (size_t i = 0; i < input.length(); i += 2) {
        istringstream iss(input.substr(i, 2));
        uint64_t temp;
        iss >> hex >> temp;
        res += static_cast<unsigned char>(temp);
    }
    return res;
}

static string hexEncodeFromFileName(const string& filename) {
    ifstream inputFile(filename, ios::binary);
    stringstream inputStream;
    inputStream << inputFile.rdbuf();
    return hexEncode(inputStream.str());
}

NapShiftForceProxy::NapShiftForceProxy() : SerializationProxy("NapShiftForce") {
}

void NapShiftForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 3);
    const NapShiftForce& force = *reinterpret_cast<const NapShiftForce*>(object);
    std::string modelType = force.getModelType();

    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setBoolProperty("useEnsembleAveraging", force.usesEnsembleAveraging());
    node.setStringProperty("modelType", force.getModelType());
    //node.setDoubleProperty("K", force.getKDefaultValue());

    SerializationNode& properties = node.createChildNode("Properties");
    for (auto& prop : force.getProperties())
        properties.createChildNode("Property").setStringProperty("name", prop.first).setStringProperty("value", prop.second);

    SerializationNode& peptides = node.createChildNode("Peptides");
    for (int i = 0; i < force.getNumPeptides(); i++) {
        if (modelType == "all_atom") {
            int N, C, CA, CB, G, D;
            char resType;
            std::map<std::string, double> csExp1, csExp2, csRC, csScale;
            int resId;
            std::string chainID;
            force.getPeptideParameters(i, N, C, CA, CB, G, D, resType, csExp1, csExp2, csRC, csScale, resId, chainID);
            std::string str_resType{resType};
            peptides.createChildNode("Peptide").setIntProperty("N", N).setDoubleProperty("C", C).setDoubleProperty("CA", CA).setDoubleProperty("CB", CB).setDoubleProperty("G", G).setDoubleProperty("D", D)
                                               .setDoubleProperty("csExp1_N", csExp1["N"]).setDoubleProperty("csExp1_C", csExp1["C"]).setDoubleProperty("csExp1_CA", csExp1["CA"]).setDoubleProperty("csExp1_CB", csExp1["CB"]).setDoubleProperty("csExp1_H", csExp1["H"]).setDoubleProperty("csExp1_HA", csExp1["HA"])
                                               .setDoubleProperty("csExp2_N", csExp2["N"]).setDoubleProperty("csExp2_C", csExp2["C"]).setDoubleProperty("csExp2_CA", csExp2["CA"]).setDoubleProperty("csExp2_CB", csExp2["CB"]).setDoubleProperty("csExp2_H", csExp2["H"]).setDoubleProperty("csExp2_HA", csExp2["HA"])
                                               .setDoubleProperty("csRC_N", csRC["N"]).setDoubleProperty("csRC_C", csRC["C"]).setDoubleProperty("csRC_CA", csRC["CA"]).setDoubleProperty("csRC_CB", csRC["CB"]).setDoubleProperty("csRC_H", csRC["H"]).setDoubleProperty("csRC_HA", csRC["HA"])
                                               .setDoubleProperty("csScale_N", csScale["N"]).setDoubleProperty("csScale_C", csScale["C"]).setDoubleProperty("csScale_CA", csScale["CA"]).setDoubleProperty("csScale_CB", csScale["CB"]).setDoubleProperty("csScale_H", csScale["H"]).setDoubleProperty("csScale_HA", csScale["HA"])
                                               .setStringProperty("resType", str_resType).setIntProperty("resId", resId).setStringProperty("chainID", chainID);
        } else {
            int bb, sc;
            char resType;
            std::map<std::string, double> csExp1, csExp2, csRC, csScale;
            int resId;
            std::string chainID;
            force.getPeptideParameters(i, bb, sc, resType, csExp1, csExp2, csRC, csScale, resId, chainID);
            std::string str_resType{resType};
            peptides.createChildNode("Peptide").setIntProperty("bb", bb).setDoubleProperty("sc", sc)
                                               .setDoubleProperty("csExp1_N", csExp1["N"]).setDoubleProperty("csExp1_C", csExp1["C"]).setDoubleProperty("csExp1_CA", csExp1["CA"]).setDoubleProperty("csExp1_CB", csExp1["CB"]).setDoubleProperty("csExp1_H", csExp1["H"]).setDoubleProperty("csExp1_HA", csExp1["HA"])
                                               .setDoubleProperty("csExp2_N", csExp2["N"]).setDoubleProperty("csExp2_C", csExp2["C"]).setDoubleProperty("csExp2_CA", csExp2["CA"]).setDoubleProperty("csExp2_CB", csExp2["CB"]).setDoubleProperty("csExp2_H", csExp2["H"]).setDoubleProperty("csExp2_HA", csExp2["HA"])
                                               .setDoubleProperty("csRC_N", csRC["N"]).setDoubleProperty("csRC_C", csRC["C"]).setDoubleProperty("csRC_CA", csRC["CA"]).setDoubleProperty("csRC_CB", csRC["CB"]).setDoubleProperty("csRC_H", csRC["H"]).setDoubleProperty("csRC_HA", csRC["HA"])
                                               .setDoubleProperty("csScale_N", csScale["N"]).setDoubleProperty("csScale_C", csScale["C"]).setDoubleProperty("csScale_CA", csScale["CA"]).setDoubleProperty("csScale_CB", csScale["CB"]).setDoubleProperty("csScale_H", csScale["H"]).setDoubleProperty("csScale_HA", csScale["HA"])
                                               .setStringProperty("resType", str_resType).setIntProperty("resId", resId).setStringProperty("chainID", chainID);
        }
    }           
}

void* NapShiftForceProxy::deserialize(const SerializationNode& node) const {
    int storedVersion = node.getIntProperty("version");
    if (storedVersion > 3)
        throw OpenMMException("Unsupported version number");
    NapShiftForce* force = new NapShiftForce();
    if (node.hasProperty("forceGroup"))
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
    if (node.hasProperty("usesPeriodic"))
        force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
    if (node.hasProperty("useEnsembleAveraging"))
        force->setUsesEnsembleAveraging(node.getBoolProperty("useEnsembleAveraging"));
    //if (node.hasProperty("K"))
    //    force->setKDefaultValue(node.getDoubleProperty("K"));
    if (node.hasProperty("modelType"))
        force->setModelType(node.getStringProperty("modelType"));

    for (const SerializationNode& child : node.getChildren()) {
        if (child.getName() == "Properties")
            for (auto& property : child.getChildren())
                force->setProperty(property.getStringProperty("name"), property.getStringProperty("value"));
        if (child.getName() == "Peptides") {
            for (auto& p : child.getChildren()) {
                std::map<std::string, double> csExp1 {{"N", -1}, {"C", -1}, {"CA", -1}, {"CB", -1}, {"H", -1}, {"HA", -1}};
                csExp1["N"]  = p.getDoubleProperty("csExp1_N");
                csExp1["C"]  = p.getDoubleProperty("csExp1_C");
                csExp1["CA"] = p.getDoubleProperty("csExp1_CA");
                csExp1["CB"] = p.getDoubleProperty("csExp1_CB");
                csExp1["H"]  = p.getDoubleProperty("csExp1_H");
                csExp1["HA"] = p.getDoubleProperty("csExp1_HA");
                std::map<std::string, double> csExp2 {{"N", -1}, {"C", -1}, {"CA", -1}, {"CB", -1}, {"H", -1}, {"HA", -1}};
                csExp2["N"]  = p.getDoubleProperty("csExp2_N");
                csExp2["C"]  = p.getDoubleProperty("csExp2_C");
                csExp2["CA"] = p.getDoubleProperty("csExp2_CA");
                csExp2["CB"] = p.getDoubleProperty("csExp2_CB");
                csExp2["H"]  = p.getDoubleProperty("csExp2_H");
                csExp2["HA"] = p.getDoubleProperty("csExp2_HA");
                std::map<std::string, double> csRC {{"N", -1}, {"C", -1}, {"CA", -1}, {"CB", -1}, {"H", -1}, {"HA", -1}};
                csRC["N"]  = p.getDoubleProperty("csRC_N");
                csRC["C"]  = p.getDoubleProperty("csRC_C");
                csRC["CA"] = p.getDoubleProperty("csRC_CA");
                csRC["CB"] = p.getDoubleProperty("csRC_CB");
                csRC["H"]  = p.getDoubleProperty("csRC_H");
                csRC["HA"] = p.getDoubleProperty("csRC_HA");
                std::map<std::string, double> csScale {{"N", -1}, {"C", -1}, {"CA", -1}, {"CB", -1}, {"H", -1}, {"HA", -1}};
                csScale["N"]  = p.getDoubleProperty("csScale_N");
                csScale["C"]  = p.getDoubleProperty("csScale_C");
                csScale["CA"] = p.getDoubleProperty("csScale_CA");
                csScale["CB"] = p.getDoubleProperty("csScale_CB");
                csScale["H"]  = p.getDoubleProperty("csScale_H");
                csScale["HA"] = p.getDoubleProperty("csScale_HA");

                if (node.getStringProperty("modelType") == "all_atom") {
                    force->addPeptide(p.getIntProperty("N"), p.getIntProperty("C"), p.getIntProperty("CA"), p.getIntProperty("CB"), p.getIntProperty("G"), p.getIntProperty("D"),
                                      p.getStringProperty("resType")[0], csExp1, csExp2, csRC, csScale, p.getIntProperty("resId"), p.getStringProperty("chainID"));
                } else {
                    force->addPeptide(p.getIntProperty("bb"), p.getIntProperty("sc"),
                                      p.getStringProperty("resType")[0], csExp1, csExp2, csRC, csScale, p.getIntProperty("resId"), p.getStringProperty("chainID"));                    
                }
            }
        }
    }
    return force;
}
