from setuptools import setup, Extension
import os
import platform

version = '1.0'
openmm_dir = ''
torch_include_dirs = '/home/mcul245/anaconda3/envs/BuildOpenMMNapShift/include;/home/mcul245/anaconda3/envs/BuildOpenMMNapShift/include/torch/csrc/api/include'.split(';')
napshift_plugin_header_dir = '/shares/mcul245/research/ressci201900070-rcha387/openmm-napshift/openmmapi/include'
napshift_plugin_library_dir = 'NAPSHIFT_PLUGIN_LIBRARY_DIR'
torch_dir, _ = os.path.split('/home/mcul245/anaconda3/envs/BuildOpenMMNapShift/lib/libtorch.so') 

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++17']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13', '-Wl', '-rpath', openmm_dir+'/lib', '-rpath', torch_dir]

extension = Extension(name='openmmnapshift._napshiftforce',
                      sources=['openmmnapshift/NapShiftForceWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMNapShift'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), napshift_plugin_header_dir] + torch_include_dirs,
                      library_dirs=[os.path.join(openmm_dir, 'lib'), napshift_plugin_library_dir],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib')],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='openmmnapshift',
      version=version,
      py_modules=['openmmnapshift.napshiftforce'],
      ext_modules=[extension],
      packages=['openmmnapshift',],
      package_data={'openmmnapshift':['PytorchModels/*.pt']}
     )
