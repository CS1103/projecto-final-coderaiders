#pragma once
#include <functional>
#include <cmath>
#include "utec/algebra/Tensor.h"

namespace utec::algebra {
    template <typename T, size_t Rank>
    class Tensor;
}
namespace utec::nn {
    using utec::algebra::Tensor;
    class Activation {
    private:
        std::function<float(float)> func_;
        std::function<float(float)> deriv_;
    public:
        Activation(std::function<float(float)> f, std::function<float(float)> df)
                : func_(std::move(f)), deriv_(std::move(df)) {}

        float operator()(float x) const noexcept {
            return func_(x);
        }
        float derivative(float x) const noexcept {
            return deriv_(x);
        }
        template<typename T>
        Tensor<T, 2> apply(const Tensor<T, 2>& input) const {
            Tensor<T, 2> out = input;
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = static_cast<T>(func_(static_cast<float>(input[i])));
            }
            return out;
        }
        template<typename T>
        Tensor<T, 2> derivative_apply(const Tensor<T, 2>& input) const {
            Tensor<T, 2> out = input;
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = static_cast<T>(deriv_(static_cast<float>(input[i])));
            }
            return out;
        }
    };
    static const Activation relu(
            [](float x) { return x > 0.0f ? x : 0.0f; },
            [](float x) { return x > 0.0f ? 1.0f : 0.0f; }
    );
    static const Activation sigmoid(
            [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
            [](float x) {
                float s = 1.0f / (1.0f + std::exp(-x));
                return s * (1.0f - s);
            }
    );
}