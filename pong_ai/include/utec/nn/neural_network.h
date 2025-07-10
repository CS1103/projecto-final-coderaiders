#pragma once
#include <vector>
#include <memory>
#include "layer.h"
#include "loss.h"
#include "dense.h"

namespace utec::nn {
    template <typename T>
    class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers;
        MSELoss<T> criterion;
    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers.push_back(std::move(layer));
        }
        Tensor<T, 2> forward(const Tensor<T, 2>& x) {
            Tensor<T, 2> out = x;
            for (auto& layer : layers)
                out = layer->forward(out);
            return out;
        }
        void backward(const Tensor<T, 2>& grad) {
            Tensor<T, 2> g = grad;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                g = (*it)->backward(g);
        }
        void optimize(T lr) {
            for (auto& layer : layers) {
                if (auto d = dynamic_cast<Dense<T>*>(layer.get()))
                    d->optimize(lr);
            }
        }
        void train(const Tensor<T, 2>& X, const Tensor<T, 2>& Y, size_t epochs, T lr) {
            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                auto pred = forward(X);
                if (pred.shape() != Y.shape()) {
                    throw std::invalid_argument("Shape mismatch: pred vs Y en train()");
                }
                auto loss = criterion.forward(pred, Y);
                auto grad = criterion.backward();
                backward(grad);
                optimize(lr);
            }
        }
    };

}