

add_executable(testmap testmap.cpp)
target_link_libraries(testmap _soapxx)
install(TARGETS testmap RUNTIME DESTINATION ${LOCAL_INSTALL_DIR}/bin)

