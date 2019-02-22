file(GLOB LOCAL_SOURCES *.cpp)

# CONFIGURE LINALG BACKEND
message("Linalg. backend ...")
find_package(GSL)
find_package(MKL)
if (TRUE)
    message(STATUS "Using MKL.")
    include_directories(${MKL_INCLUDE_DIRS})
    set(LINALG_LIBRARIES ${MKL_LIBRARIES})
    file(GLOB LINALG_BACKEND_SOURCES mkl/*.cpp)
elseif (GSL_FOUND)
    message(STATUS "Using GSL.")
    include_directories(${GSL_INCLUDE_DIRS})
    set(LINALG_LIBRARIES ${GSL_LIBRARIES})
    file(GLOB LINALG_BACKEND_SOURCES gsl/*.cpp)
else()
    message(FATAL_ERROR "Neither gsl nor mkl found.")
endif()
set(LOCAL_SOURCES ${LOCAL_SOURCES} ${LINALG_BACKEND_SOURCES})

# SUMMARIZE SOURCES
message("Sources: soap/linalg")
foreach(item ${LOCAL_SOURCES})
    message(STATUS " o ${item}")
endforeach()

# ADD & LINK LIBRARY
add_library(_linalg ${LOCAL_SOURCES})
target_link_libraries(_linalg ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${MPI_LIBRARIES} ${LINALG_LIBRARIES})
set_target_properties(_linalg PROPERTIES PREFIX "" SUFFIX ".so" LIBRARY_OUTPUT_DIRECTORY .)

# INSTALL
install(TARGETS _linalg LIBRARY DESTINATION ${LOCAL_INSTALL_DIR}/linalg)
install(FILES __init__.py DESTINATION ${LOCAL_INSTALL_DIR}/linalg)
#install(DIRECTORY linalg DESTINATION ${LOCAL_INSTALL_DIR})

# HEADERS
file(GLOB HEADERS *.hpp)
install(FILES ${HEADERS} DESTINATION ${LOCAL_INSTALL_DIR}/include/soap/linalg)
