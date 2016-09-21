# CONFIGURE CMAKE
message("CMake version: ${CMAKE_VERSION}")
cmake_minimum_required(VERSION 2.8.3)

# PROJECT OPTIONS
project(soapxx)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(LOCAL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/soap)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# BUILD OPTIONS
enable_language(CXX)
#set(CMAKE_CXX_COMPILER "/usr/local/shared/intel/compilers_and_libraries_2016.2.181/linux/bin/intel64/icc")
message("C++ compiler: " ${CMAKE_CXX_COMPILER} " " ${CMAKE_CXX_COMPILER_ID})
option(BUILD_SHARED_LIBS "Build shared libs" ON)
if(${CMAKE_VERSION} VERSION_GREATER 3.1)
    message("Setting C++ standard 11 (CMake version > 3.1)")
    set(CMAKE_CXX_STANDARD 11)
else(${CMAKE_VERSION} VERSION_GREATER 3.1)
    message("Setting C++ standard 11 (CMake version <= 3.1)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(${CMAKE_VERSION} VERSION_GREATER 3.1)

# LOCAL PACKAGES
# ...
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# FIND PACKAGES
find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})

if(DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
    message("-- BOOST_ROOT is set: ${BOOST_ROOT}")
else(DEFINED ENV{BOOST_ROOT})
    message("-- Note: BOOST_ROOT not set.")
endif(DEFINED ENV{BOOST_ROOT})
message("-- BOOST_ROOT is set: ${BOOST_ROOT}")
find_package(Boost 1.60.0 COMPONENTS python mpi filesystem serialization)
include_directories(${Boost_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(SOAPXX_LINK_LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${MPI_LIBRARIES} ${GSL_LIBRARIES})

# SUMMARIZE INCLUDES & LIBS
get_property(local_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
get_property(local_libs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY SOAPXX_LINK_LIBRARIES)

message(STATUS "Include directories: ")
foreach(dir ${local_dirs})
  message(STATUS " o ${dir}")
endforeach()
message(STATUS "Linked libraries:    ")
foreach(lib ${local_dirs})
  message(STATUS " o ${lib}")
endforeach()

# SUBDIRECTORIES
add_subdirectory(src/soap)

