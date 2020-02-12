file(GLOB LOCAL_SOURCES *.cpp)
pybind11_add_module(_soapgto ${LOCAL_SOURCES})
install(TARGETS _soapgto LIBRARY DESTINATION ${LOCAL_INSTALL_DIR}/external)
install(FILES __init__.py DESTINATION ${LOCAL_INSTALL_DIR}/external)
