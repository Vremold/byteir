set(brt_all_includes brt_common brt_framework)

# cutlass
include_directories("${CUTLASS_ROOT}/include")
include_directories("${CUTLASS_ROOT}/tools/util/include")
# CUDA provide
file(GLOB_RECURSE brt_cuda_provider_srcs CONFIGURE_DEPENDS
  "${BRT_INCLUDE_DIR}/brt/backends/cuda/providers/default/*.h"
  "${BRT_INCLUDE_DIR}/fastertransformer_v3/includes/*.h"
  "${BRT_INCLUDE_DIR}/fastertransformer_v3/includes/*.cuh"
  "${BRT_INCLUDE_DIR}/fastertransformer_v4/includes/*.h"
  "${BRT_INCLUDE_DIR}/fastertransformer_v4/includes/*.cuh"
  "${LIB_ROOT}/backends/cuda/providers/default/*.h"
  "${LIB_ROOT}/backends/cuda/providers/default/*.cc"
  "${LIB_ROOT}/backends/cuda/providers/default/*.cu"
)

list(APPEND brt_all_providers_srcs ${brt_cuda_provider_srcs})
list(APPEND brt_all_includes brt_device_cuda)

source_group(TREE ${REPO_ROOT} FILES ${brt_cuda_provider_srcs})

brt_add_object_library(brt_provider_cuda ${brt_cuda_provider_srcs})
target_link_libraries(brt_provider_cuda brt_framework brt_common brt_ir)
target_link_libraries(brt_provider_cuda ${BRT_CUDA_LIBRARIES})
brt_add_include_to_target(brt_provider_cuda ${brt_all_includes})
set_target_properties(brt_provider_cuda PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(brt_provider_cuda PROPERTIES FOLDER "Brt")

install(
  DIRECTORY "${BRT_INCLUDE_DIR}/brt/backends/cuda/providers"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/brt/backends/cuda")