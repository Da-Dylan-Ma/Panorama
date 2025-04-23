cmake_minimum_required(VERSION 3.12)
project(DisplayImage)

# ---- OpenCV ----
set(OpenCV_DIR ~/local-opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

# ---- OpenMP ----
find_package(OpenMP REQUIRED)        # adds OpenMP::OpenMP_CXX target
if (OpenMP_FOUND)
    message(STATUS "OpenMP found")
endif()

# ---- Eigen ----
#find_package(Eigen3 REQUIRED)

# ---- compile flags you probably want for every target ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-O3 -march=native)   # optimise & vectorise

# ---- helper macro to link common libs ----
function(link_all target)
    target_link_libraries(${target}
            ${OpenCV_LIBS}
#            Eigen3::Eigen
            OpenMP::OpenMP_CXX)          # <<< add OpenMP everywhere
endfunction()

# ---- executables ----
add_executable(MSOP               MSOP.cpp)
add_executable(blending_direct    blending_direct.cpp)
add_executable(blending_alpha     blending_alpha.cpp)
add_executable(blending_graphcut  blending_graphcut.cpp)
add_executable(blending_poisson   blending_poisson.cpp)
add_executable(test blending_photomontage.cpp)

link_all(MSOP)
link_all(blending_direct)
link_all(blending_alpha)
link_all(blending_graphcut)
link_all(blending_poisson)
link_all(test)
