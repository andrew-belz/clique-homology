cmake_minimum_required(VERSION 3.24)
project(graphoplib VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

set(TARGET graphoplib)

set(HEADERS 
    src/internal/betti.hpp
    src/internal/boundary_maps.hpp
    src/internal/data_types.hpp
    src/internal/graph.hpp
    src/internal/matrix.hpp
    src/internal/sparse_matrix.hpp
    src/internal/maximal_cliques.hpp
    src/internal/permutation_test.hpp
    src/internal/pval_vector.hpp
    src/internal/pval.hpp
    src/internal/validation.hpp
)

set(SOURCE
    src/api/betti.cpp
    src/core/validation.cpp
    src/graph/graph.cpp
    src/graph/permuted_graph.cpp
    src/internal/maximal_cliques.cpp
    src/main/betti.cpp
    src/stats/permutation_test.cpp
    src/stats/pval.cpp
    src/stats/pval_vector.cpp
    src/topology/boundary_maps.cpp
    src/topology/matrix.cpp
    src/topology/sparse_matrix.cpp
)

add_library(${TARGET} STATIC ${SOURCE} ${HEADERS})

target_include_directories(${TARGET}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/internal
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
    GIT_TAG         3.4.0
)
FetchContent_MakeAvailable(Eigen3)

set(BUILD_TESTING ON CACHE BOOL "" FORCE)

target_link_libraries(graphoplib PUBLIC Eigen3::Eigen)

add_executable(test_betti tests/test_betti.cpp)
target_link_libraries(test_betti PRIVATE ${TARGET} GTest::gtest_main)

add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph PRIVATE ${TARGET} GTest::gtest_main)

add_executable(test_maximal_cliques tests/test_maximal_cliques.cpp)
target_link_libraries(test_maximal_cliques PRIVATE ${TARGET} GTest::gtest_main)

# add_executable(test_matrix tests/test_matrix.cpp)
# target_link_libraries(test_matrix PRIVATE ${TARGET} GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_betti)
gtest_discover_tests(test_graph)
gtest_discover_tests(test_maximal_cliques)
# gtest_discover_tests(test_matrix)