#
# excoder



set(EXCODER_TARGET excoder)
add_executable(${EXCODER_TARGET} main.cpp excoder.cpp)

target_link_libraries(${EXCODER_TARGET} PRIVATE ggml common common-ggml)

#
# excoder-quantize

# set(EXCODER_QT_TARGET excoder-quantize)
# add_executable(${EXCODER_QT_TARGET} quantize.cpp)

# target_link_libraries(${EXCODER_QT_TARGET} PRIVATE ggml common common-ggml)

# 
# excoder-serve

set(EXCODER_SERVE_TARGET excoder-serve)
add_executable(${EXCODER_SERVE_TARGET} serve.cpp excoder.cpp)
target_link_libraries(${EXCODER_SERVE_TARGET} PRIVATE ggml common common-ggml)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

if (GGML_STATIC)
    #add_link_options(-static-libgcc -static-libstdc++)
    SET(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc -static-libstdc++")
    SET_TARGET_PROPERTIES(${EXCODER_TARGET} PROPERTIES 
LINK_SEARCH_END_STATIC 1)
    SET_TARGET_PROPERTIES(${EXCODER_QT_TARGET} PROPERTIES 
LINK_SEARCH_END_STATIC 1)
SET_TARGET_PROPERTIES(${EXCODER_SERVE_TARGET} PROPERTIES 
LINK_SEARCH_END_STATIC 1)
endif()