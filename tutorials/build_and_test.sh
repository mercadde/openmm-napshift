conda activate BuildOpenMMNapShift
cd ../build
make install
make PythonInstall
cd python
python -m pip install .
cd ../../tutorials
python ./test_cpp_side_replicas.py $@
