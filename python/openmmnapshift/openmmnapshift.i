%pythonbegin %{
import sys
if sys.platform == 'win32':
    import os
    import torch
    import openmm
    openmmnapshift_library_path = openmm.version.openmm_library_path

    _path = os.environ['PATH']
    os.environ['PATH'] = r'%(lib)s;%(lib)s\plugins;%(path)s' % {'lib': openmmnapshift_library_path, 'path': _path}

    os.add_dll_directory(openmmnapshift_library_path)

%}

%module napshiftforce

%include "factory.i"
%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_map.i>
%include <std_vector.i>

%{
#include "NapShiftForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/serialization/import.h>
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

%typemap(in) const torch::jit::Module&(torch::jit::Module mod) {
    py::object o = py::reinterpret_borrow<py::object>($input);
    py::object pybuffer = py::module::import("io").attr("BytesIO")();
    py::module::import("torch.jit").attr("save")(o, pybuffer);
    std::string s = py::cast<std::string>(pybuffer.attr("getvalue")());
    std::stringstream buffer(s);
    mod = torch::jit::load(buffer);
    $1 = &mod;
}

%typemap(out) const torch::jit::Module& {
    std::stringstream buffer;
    $1->save(buffer);
    auto pybuffer = py::module::import("io").attr("BytesIO")(py::bytes(buffer.str()));
    $result = py::module::import("torch.jit").attr("load")(pybuffer).release().ptr();
}

%typecheck(SWIG_TYPECHECK_POINTER) const torch::jit::Module& {
    py::object o = py::reinterpret_borrow<py::object>($input);
    py::handle ScriptModule = py::module::import("torch.jit").attr("ScriptModule");
    $1 = py::isinstance(o, ScriptModule);
}

namespace std {
    %template(property_map) map<string, string>;
    %template(csExp_map) map<string, double>;
    %template(atomVector) vector<string>;
}

namespace NapShiftPlugin {

class NapShiftForce : public OpenMM::Force {
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
    void setProperty(const std::string& name, const std::string& value);
    const std::map<std::string, std::string>& getProperties() const;
    void setModelType(std::string modelType);
    std::string getModelType() const;
    double getKDefaultValue() const;
    void setKDefaultValue(double defaultK);
    std::string getPytorchModelsDir() const;
    void setPytorchModelsDir(std::string pytorchModelsDir);
    /*
     * Add methods for casting a Force to a NapShiftForce.
    */
    %extend {
        static NapShiftPlugin::NapShiftForce& cast(OpenMM::Force& force) {
            return dynamic_cast<NapShiftPlugin::NapShiftForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<NapShiftPlugin::NapShiftForce*>(&force) != NULL);
        }
    }
};

}
