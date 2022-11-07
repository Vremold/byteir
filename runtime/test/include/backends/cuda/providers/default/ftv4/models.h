
#pragma once

#include "brt/core/ir/builder.h"

namespace brt {
namespace test {
namespace cuda {
namespace ftv4 {

const void *CreateLayerNorm(brt::ir::ByREBuilder &byre_builder);

const void *CreateLayerNormBackward(brt::ir::ByREBuilder &byre_builder);

const void *CreateSoftmax(brt::ir::ByREBuilder &byre_builder);

const void *CreateSoftmaxBackward(brt::ir::ByREBuilder &byre_builder);

const void *CreateLinearGeluDropout(brt::ir::ByREBuilder &byre_builder);

const void *CreateLinearGeluDropoutBackward(brt::ir::ByREBuilder &byre_builder);

const void *CreateLinearTranspose0213(brt::ir::ByREBuilder &byre_builder);

const void *
CreateLinearTranspose0213Backward(brt::ir::ByREBuilder &byre_builder);

const void *CreateTranspose4d2013(brt::ir::ByREBuilder &byre_builder);

const void *CreateMatmul(brt::ir::ByREBuilder &byre_builder);

} // namespace ftv4
} // namespace cuda
} // namespace test
} // namespace brt
