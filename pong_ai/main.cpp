#include <iostream>
#include <memory>
#include "utec/agent/EnvGym.h"
#include "utec/agent/PongAgent.h"
#include "utec/nn/dense.h"

int main() {
    using T = float;
    std::unique_ptr<utec::nn::ILayer<T>> modelo =
            std::make_unique<utec::nn::Dense<T>>(3, 1);
    utec::nn::PongAgent<T> agente(std::move(modelo));
    utec::nn::EnvGym env;
    float reward = 0;
    bool done = false;
    utec::nn::State s = env.reset();
    std::cout << "\n Test mínimo de integración: Dense + Agent + EnvGym\n";
    for (int paso = 0; paso < 5; ++paso) {
        std::cout << "\nPaso " << paso << ":\n";
        int accion = agente.act(s);
        s = env.step(accion, reward, done);

        std::cout << "Acción: " << accion
                  << " | Recompensa: " << reward
                  << " | Terminado: " << std::boolalpha << done << "\n";
        if (done) {
            std::cout << " Episodio finalizado.\n";
            break;
        }
    }
    return 0;
}

