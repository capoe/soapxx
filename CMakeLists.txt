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
find_package(PythonInterp REQUIRED)
set(Python_ADDITIONAL_VERSIONS "$ENV{PYTHON_VERSION_STRING}")
message(PYTHON_ADD_VERSIONS= ${Python_ADDITIONAL_VERSIONS})
find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})
message(PYTHON_INC_PATH= ${PYTHON_INCLUDE_DIRS})
message(PYTHON_LIB_PATH= ${PYTHON_LIBRARIES})
message(PYTHON_LIB_STR= ${PYTHONLIBS_VERSION_STRING})
string(REGEX REPLACE "libpython.*" "" LIBPY_ROOT_PATH ${PYTHON_LIBRARIES})
message(PYTHON_LIB_ROOT= ${LIBPY_ROOT_PATH})

find_package(Numpy)
if(NUMPY_FOUND)
    string(CONCAT NUMPY_INCLUDE_NUMPYDIR ${NUMPY_INCLUDE_DIR} "/numpy/")
    file(TO_NATIVE_PATH ${NUMPY_INCLUDE_NUMPYDIR} NUMPY_INCLUDE_ADDDIR)
    include_directories(${NUMPY_INCLUDE_DIR})
    include_directories(${NUMPY_INCLUDE_ADDDIR})
endif()
message(NUMPY_INC_DIR= ${NUMPY_INCLUDE_DIR})
message(NUMPY_INCLUDE_ADD= ${NUMPY_INCLUDE_ADDDIR})

if(DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
    message("-- BOOST_ROOT is set: ${BOOST_ROOT}")
else(DEFINED ENV{BOOST_ROOT})
    message("-- Note: BOOST_ROOT not set. Let's see whether we can find BOOST anyway.")
endif(DEFINED ENV{BOOST_ROOT})

find_package(Boost REQUIRED)
if(Boost_FOUND)
  set(BOOST_MAJOR_MINOR_VERSION "$ENV{Boost_MAJOR_VERSION}.$ENV{Boost_MINOR_VERSION}")
  if(BOOST_MAJOR_MINOR_VERSION VERSION_GREATER 1.63)
    if (APPLE OR WIN32) # Windows & Homebrew convention
        set(TMP_BOOST_PYTHON_COMPONENT "python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}" "numpy${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}" "mpi" "filesystem" "serialization")
    else() # Debian convention
        set(TMP_BOOST_PYTHON_COMPONENT "python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}" "numpy${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}" "mpi" "filesystem" "serialization")
    endif()
  else()
    set(TMP_BOOST_PYTHON_COMPONENT "python" "mpi" "filesystem" "serialization")
  endif()
else()
  set(TMP_BOOST_PYTHON_COMPONENT "python" "mpi" "filesystem" "serialization")
endif()
find_package(Boost COMPONENTS ${TMP_BOOST_PYTHON_COMPONENT} REQUIRED)
unset(TMP_BOOST_PYTHON_COMPONENT)

#find_package(Boost COMPONENTS python python3 mpi filesystem serialization)
include_directories(${Boost_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(SOAPXX_LINK_LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${MPI_LIBRARIES} ${GSL_LIBRARIES} ${NUMPY_LIBRARIES})

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

