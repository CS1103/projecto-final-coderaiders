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
    utec::nn::EnvGym entorno;
    float recompensa = 0.0f;
    bool terminado = false;
    utec::nn::State estado = entorno.reset();
    std::cout << " Iniciando simulación Pong con red, agente y entorno...\n";
    for (int paso = 0; paso < 10; ++paso) {
        std::cout << "\nPaso " << paso << ":\n";
        std::cout << "Estado actual → bola: (" << estado.ball_x << ", " << estado.ball_y
                  << "), paleta: " << estado.paddle_y << "\n";
        int accion = agente.act(estado);
        estado = entorno.step(accion, recompensa, terminado);
        std::cout << "Acción tomada: " << accion << "\n";
        std::cout << "Recompensa recibida: " << recompensa << "\n";
        std::cout << "¿Terminado? " << std::boolalpha << terminado << "\n";
        if (terminado) {
            std::cout << " Episodio finalizado en paso " << paso << "\n";
            break;
        }
    }
    return 0;
}