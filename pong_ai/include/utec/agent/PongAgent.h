#pragma once
#include "utec/algebra/Tensor.h"
#include "utec/nn/layer.h"
#include "utec/agent/EnvGym.h"
#include <memory>

namespace utec {
    namespace nn {
        template <typename T>
        class PongAgent {
        private:
            std::unique_ptr<ILayer<T>> model_;
        public:
            explicit PongAgent(std::unique_ptr<ILayer<T>> model);
            int act(const State& s);
            void learn_on_policy(
                    const State& s,
                    int a,
                    float r,
                    const State& s_next,
                    float gamma,
                    float lr
            );
        };
    }
}