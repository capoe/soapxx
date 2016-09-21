file(GLOB LOCAL_SOURCES *.cpp)

# SUMMARIZE SOURCES
message("Sources: soap/linalg/permanent")
foreach(item ${LOCAL_SOURCES})
    message(STATUS " o ${item}")
endforeach()

# ADD & LINK LIBRARY
add_library(_permanent ${LOCAL_SOURCES})
target_compile_options(_permanent PRIVATE "-O3" PRIVATE "-Ofast" PRIVATE "-march=native")
target_link_libraries(_permanent ${PYTHON_LIBRARIES})
set_target_properties(_permanent PROPERTIES PREFIX "" SUFFIX ".so" LIBRARY_OUTPUT_DIRECTORY .)

# INSTALL
install(TARGETS _permanent LIBRARY DESTINATION ${LOCAL_INSTALL_DIR}/linalg/permanent)
install(FILES __init__.py DESTINATION ${LOCAL_INSTALL_DIR}/linalg/permanent)

