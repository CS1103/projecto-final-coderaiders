#pragma once
#include "../algebra/Tensor.h"
#include "layer.h"
#include <random>

namespace utec::nn {
    using utec::algebra::Tensor;
    template <typename T>
    class Dense : public ILayer<T> {
    private:
        Tensor<T, 2> W, dW;
        Tensor<T, 1> b, db;
        Tensor<T, 2> last_x;
    public:
        Dense(size_t in_feats, size_t out_feats)
                : W({in_feats, out_feats}),
                  dW({in_feats, out_feats}),
                  b({out_feats}),
                  db({out_feats}) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(-1.0, 1.0);
            for (auto& w : W) w = dist(gen);
            for (auto& bi : b) bi = dist(gen);
        }
        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            last_x = x;
            auto y = x * W;
            for (size_t i = 0; i < y.shape()[0]; ++i)
                for (size_t j = 0; j < y.shape()[1]; ++j)
                    y(i, j) += b(j);
            return y;
        }
        Tensor<T, 2> backward(const Tensor<T, 2>& grad) override {
            auto grad_input = grad * W.transpose_2d();
            dW = last_x.transpose_2d() * grad;

            for (size_t j = 0; j < db.shape()[0]; ++j) {
                db(j) = 0;
                for (size_t i = 0; i < grad.shape()[0]; ++i)
                    db(j) += grad(i, j);
            }

            return grad_input;
        }
        void optimize(T lr) {
            for (size_t i = 0; i < W.shape()[0]; ++i)
                for (size_t j = 0; j < W.shape()[1]; ++j)
                    W(i, j) -= lr * dW(i, j);
            for (size_t j = 0; j < b.shape()[0]; ++j)
                b(j) -= lr * db(j);
        }
    };
}