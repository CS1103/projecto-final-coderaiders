#include <iostream>
#include "utec/algebra/Tensor.h"
#include "utec/nn/neural_network.h"
#include "utec/nn/dense.h"

int main() {
    using T = float;
    utec::algebra::Tensor<T, 2> X(2, 2);
    X = {
            0.0f, 0.0f,
            1.0f, 1.0f
    };
    utec::algebra::Tensor<T, 2> Y(2, 2);
    Y = {
            0.0f, 1.0f,
            1.0f, 0.0f
    };
    utec::nn::NeuralNetwork<T> net;
    net.add_layer(std::make_unique<utec::nn::Dense<T>>(2, 2));
    net.add_layer(std::make_unique<utec::nn::Dense<T>>(2, 2));
    std::cout << "Entrenando...\n";
    try {
        net.train(X, Y, 1000, 0.1f);
        std::cout << "Entrenamiento terminado.\n";
        auto pred = net.forward(X);
        std::cout << "Pred shape : " << pred.shape()[0]
                  << " × "          << pred.shape()[1] << "\n";
        std::cout << "Real shape : " << Y.shape()[0]
                  << " × "          << Y.shape()[1] << "\n\n";
        std::cout << "Predicciones:\n" << pred << "\n";
    }
    catch (const std::invalid_argument& e) {
        std::cerr << "Error atrapado: " << e.what() << "\n";
    }
    return 0;
}