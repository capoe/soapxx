# COMPILE LIBRARIES
set(LD_LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${MPI_LIBRARIES} ${SOAP_LIBRARIES} ${GTEST_LIBRARIES})

add_executable(test_functions.exe test_functions.cpp)
target_link_libraries(test_functions.exe ${LD_LIBRARIES})
install(TARGETS test_functions.exe DESTINATION ${LOCAL_INSTALL_DIR})

add_executable(test_dylm.exe test_dylm.cpp)
target_link_libraries(test_dylm.exe ${LD_LIBRARIES})
install(TARGETS test_dylm.exe DESTINATION ${LOCAL_INSTALL_DIR})

file(GLOB local_sources gtest_*.cpp)

add_executable(test.exe ${local_sources})
target_link_libraries(test.exe ${LD_LIBRARIES})
install(TARGETS test.exe DESTINATION ${LOCAL_INSTALL_DIR})

