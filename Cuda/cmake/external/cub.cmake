include(ExternalProject)

set(CUB_PREFIX_DIR ${THIRD_PARTY_PATH}/cub)
set(CUB_SOURCE_DIR ${CUB_PREFIX_DIR}/src/extern_cub)
set(CUB_REPOSITORY https://github.com/NVIDIA/cub.git)
set(CUB_TAG 2.1.0)

cache_third_party(
  extern_cub
  REPOSITORY
  ${CUB_REPOSITORY}
  TAG
  ${CUB_TAG}
  DIR
  CUB_SOURCE_DIR)

set(CUB_INCLUDE_DIR ${CUB_SOURCE_DIR}/include)
include_directories(${CUB_INCLUDE_DIR})

ExternalProject_Add(
  extern_cub
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${CUB_DOWNLOAD_CMD}"
  PREFIX ${CUB_PREFIX_DIR}
  SOURCE_DIR ${CUB_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(cub INTERFACE)
add_dependencies(cub extern_cub)
