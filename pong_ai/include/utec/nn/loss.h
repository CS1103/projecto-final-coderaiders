#pragma once
#include <cstddef>
#include "utec/algebra/Tensor.h"

namespace utec::algebra {
    template <typename T, size_t Rank>
    class Tensor;
}
namespace utec::nn {
    using utec::algebra::Tensor;

    template<typename T>
    class MSELoss {
    private:
        Tensor<T, 2> pred_, tar_;
    public:
        T forward(const Tensor<T, 2>& p, const Tensor<T, 2>& t) {
            if (p.shape() != t.shape()) {
                throw std::invalid_argument("Las formas de predicción y objetivo no coinciden");
            }
            pred_ = p;
            tar_ = t;
            auto d = p - t;
            auto sq = d * d;
            T sum = 0;
            for (const auto& v : sq)
                sum += v;
            return sum / static_cast<T>(sq.size());
        }

        Tensor<T, 2> backward() {
            auto g = pred_ - tar_;
            T f = static_cast<T>(2.0) / static_cast<T>(g.size());
            for (auto& v : g)
                v *= f;
            return g;
        }
    };
}