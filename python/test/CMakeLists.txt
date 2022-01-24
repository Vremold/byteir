add_custom_target(check-byteir-python
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH="${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}" ${Python3_EXECUTABLE} -m pytest
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ByteIRPythonModule
)