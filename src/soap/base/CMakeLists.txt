file(GLOB HEADERS *.hpp)
file(GLOB LOCAL_SOURCES *.cpp)
install(FILES ${HEADERS} DESTINATION ${LOCAL_INSTALL_DIR}/include/soap/base)

