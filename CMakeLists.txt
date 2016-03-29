# CONFIGURE CMAKE
cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(LOCAL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/soap)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# BUILD OPTIONS
option(BUILD_SHARED_LIBS "Build shared libs" ON)

# LOCAL PACKAGES
#set(BOOST_ROOT "/home/cp605/packages/install_boost/boost_1_60_0/libboost")
#set(ESPP_ROOT "/home/cp605/nomad/espressopp")
#set(ESPP_LIBRARIES ${ESPP_ROOT}/_espressopp.so)
#include_directories(${ESPP_INCLUDE_DIRS})

# FIND PACKAGES
find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Boost COMPONENTS python mpi filesystem serialization)
include_directories(${Boost_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(FFTW3 REQUIRED)
include_directories(${FFTW3_INCLUDES})

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

#link_libraries(${ESPP_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})

# SOME FRIENDLY MESSAGING
get_property(local_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
get_property(local_libs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY LINK_LIBRARIES)

message(STATUS "Include directories: ")
foreach(dir ${local_dirs})
  message(STATUS " o ${dir}")
endforeach()
message(STATUS "Linked libraries:    ")
foreach(lib ${local_dirs})
  message(STATUS " o ${lib}")
endforeach()

#message(STATUS "Espresso++ lib:     ")
#message(STATUS " o ${ESPP_LIBRARIES}")

# SUBDIRECTORIES
add_subdirectory(src)

