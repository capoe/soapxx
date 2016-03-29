file(GLOB_RECURSE LOCAL_SOURCES *.cpp)

add_library(_linalg ${LOCAL_SOURCES})
target_link_libraries(_linalg ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${MPI_LIBRARIES} ${GSL_LIBRARIES})
set_target_properties(_linalg PROPERTIES PREFIX "" SUFFIX ".so" LIBRARY_OUTPUT_DIRECTORY .)

install(TARGETS _linalg LIBRARY DESTINATION ${LOCAL_INSTALL_DIR}/linalg)
install(FILES __init__.py DESTINATION ${LOCAL_INSTALL_DIR}/linalg)
#install(DIRECTORY linalg DESTINATION ${LOCAL_INSTALL_DIR})

