#pragma once
#include "../algebra/Tensor.h"

namespace utec::nn {
    using utec::algebra::Tensor;
    template <typename T>
    class ILayer {
    public:
        virtual ~ILayer() = default;
        virtual Tensor<T, 2> forward(const Tensor<T, 2>& x) = 0;
        virtual Tensor<T, 2> backward(const Tensor<T, 2>& grad) = 0;
    };

}