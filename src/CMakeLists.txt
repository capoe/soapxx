file(GLOB_RECURSE LOCAL_SOURCES *.cpp)

add_library(_soapxx ${LOCAL_SOURCES})
target_link_libraries(_soapxx ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${MPI_LIBRARIES} ${GSL_LIBRARIES})
set_target_properties(_soapxx PROPERTIES PREFIX "" SUFFIX ".so" LIBRARY_OUTPUT_DIRECTORY .)

configure_file(SOAPRC.in SOAPRC @ONLY)

install(TARGETS _soapxx LIBRARY DESTINATION ${LOCAL_INSTALL_DIR})
install(FILES __init__.py ${CMAKE_CURRENT_BINARY_DIR}/SOAPRC DESTINATION ${LOCAL_INSTALL_DIR})
#install(DIRECTORY linalg DESTINATION ${LOCAL_INSTALL_DIR})

add_subdirectory(linalg)
add_subdirectory(tools)

