
#pragma once
namespace brt {

class KernelRegistry;

namespace cuda {

void RegisterFasterTransformerOps(KernelRegistry *registry);

} // namespace cuda
} // namespace brt
