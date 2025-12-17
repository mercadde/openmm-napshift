file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/openmmnapshift)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/python/openmmnapshift/PytorchModels)

file(GLOB_RECURSE PYTORCH_FILES RELATIVE "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/python/openmmnapshift/PytorchModels/*.pt"
)
set(COPIED_PYTORCH_FILES)
foreach(PYTORCH_FILE ${PYTORCH_FILES})
    set(infile "${CMAKE_SOURCE_DIR}/${PYTORCH_FILE}")
    set(outfile "${CMAKE_BINARY_DIR}/${PYTORCH_FILE}")
    add_custom_command(
        OUTPUT "${outfile}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${infile}" "${outfile}"
        DEPENDS "${infile}"
        COMMENT "CMake-copying file ${infile} to ${outfile}")
    set(COPIED_PYTORCH_FILES ${COPIED_PYTORCH_FILES} "${outfile}")
endforeach()
add_custom_target(CopyPytorchFiles ALL DEPENDS ${COPIED_PYTORCH_FILES})

set(WRAP_FILE openmmnapshift/NapShiftForceWrapper.cpp)
set(MODULE_NAME napshiftforce)

# Execute SWIG to generate source code for the Python module.
foreach(dir ${TORCH_INCLUDE_DIRS})
    set(torchincs "${torchincs}" "-I${dir}")
endforeach()

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
	${torchincs}
        "${CMAKE_CURRENT_SOURCE_DIR}/openmmnapshift/openmmnapshift.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/openmmnapshift/openmmnapshift.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)


# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}"  "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(NAPSHIFT_PLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(NAPSHIFT_PLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/openmmnapshift/utils.py ${CMAKE_CURRENT_BINARY_DIR}/openmmnapshift/utils.py)
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
