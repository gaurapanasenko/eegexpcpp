cmake_minimum_required(VERSION 3.18)
project(eegexpcpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "${CMAKE_SOURCE_DIR}/libtorch")
  list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
endif()

find_package(Torch REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
  URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  doctest
  URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.tar.gz
  URL_HASH SHA256=632ed2c05a7f53fa961381497bf8069093f0d6628c5f26286161fbd32a560186
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_Populate(doctest)
endif()
add_library(doctest INTERFACE)
target_include_directories(
  doctest
  INTERFACE "${doctest_SOURCE_DIR}/doctest"
)
add_library(doctest_with_main STATIC
  "${doctest_SOURCE_DIR}/doctest/parts/doctest.cpp"
)
target_link_libraries(doctest_with_main PUBLIC doctest)
target_compile_definitions(
  doctest_with_main
  PUBLIC DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
)
add_library(doctest::doctest_with_main ALIAS doctest_with_main)

add_library(
  eegexpcpp_model
  src/mlp.cpp
  src/config.cpp
  src/encoder.cpp
  src/ode.cpp
  src/ode_sindy.cpp
  src/sindy_library.cpp
  src/sindy_table.cpp
  src/decoder.cpp
  src/latent_ae_ode.cpp
  src/checkpoint.cpp
  src/dataset_h5.cpp
)
target_include_directories(
  eegexpcpp_model
  PUBLIC "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(
  eegexpcpp_model
  PUBLIC "${TORCH_LIBRARIES}" HDF5::HDF5 nlohmann_json::nlohmann_json
)
target_compile_options(
  eegexpcpp_model
  PUBLIC ${TORCH_CXX_FLAGS}
)
set_property(TARGET eegexpcpp_model PROPERTY CXX_STANDARD 20)
set_target_properties(eegexpcpp_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(eegexpcpp src/main.cpp)
target_link_libraries(eegexpcpp PRIVATE eegexpcpp_model)
target_compile_options(eegexpcpp PRIVATE ${TORCH_CXX_FLAGS})

add_executable(eegexpcpp_sindy_csv src/sindy_csv_main.cpp)
target_link_libraries(eegexpcpp_sindy_csv PRIVATE eegexpcpp_model)
target_compile_options(eegexpcpp_sindy_csv PRIVATE ${TORCH_CXX_FLAGS})

add_executable(eegexpcpp_train src/train_main.cpp)
target_link_libraries(eegexpcpp_train PRIVATE eegexpcpp_model)
target_compile_options(eegexpcpp_train PRIVATE ${TORCH_CXX_FLAGS})

add_executable(eegexpcpp_test src/latent_ae_ode_test.cpp)
target_link_libraries(
  eegexpcpp_test
  PRIVATE eegexpcpp_model doctest::doctest_with_main
)
target_compile_options(eegexpcpp_test PRIVATE ${TORCH_CXX_FLAGS})

add_executable(eegexpcpp_calculus_test src/calculus_rk4_test.cpp)
target_include_directories(
  eegexpcpp_calculus_test
  PRIVATE "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(
  eegexpcpp_calculus_test
  PRIVATE doctest::doctest_with_main
)
target_compile_features(eegexpcpp_calculus_test PRIVATE cxx_std_20)

add_library(eegexpcpp_infer SHARED src/infer_api.cpp)
target_link_libraries(eegexpcpp_infer PRIVATE eegexpcpp_model)
target_compile_options(eegexpcpp_infer PRIVATE ${TORCH_CXX_FLAGS})
set_target_properties(
  eegexpcpp_infer
  PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../libtorch/lib"
)

enable_testing()
add_test(NAME latent_ae_ode COMMAND eegexpcpp_test)
add_test(NAME calculus_rk4 COMMAND eegexpcpp_calculus_test)
