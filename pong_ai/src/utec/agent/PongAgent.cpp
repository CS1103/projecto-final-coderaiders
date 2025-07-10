#include "utec/agent/PongAgent.h"
#include "utec/algebra/Tensor.h"

namespace utec::nn {
    template <typename T>
    PongAgent<T>::PongAgent(std::unique_ptr<ILayer<T>> model)
            : model_(std::move(model)) {}
    template <typename T>
    int PongAgent<T>::act(const State& s) {
        Tensor<T, 2> input(1, 3);
        input(0, 0) = static_cast<T>(s.ball_x);
        input(0, 1) = static_cast<T>(s.ball_y);
        input(0, 2) = static_cast<T>(s.paddle_y);
        Tensor<T, 2> output = model_->forward(input);
        T value = output(0, 0);
        if (value > static_cast<T>(0.1)) return +1;
        if (value < static_cast<T>(-0.1)) return -1;
        return 0;
    }
    template <typename T>
    void PongAgent<T>::learn_on_policy(const State& s, int action,
                                       float reward, const State& s_next,
                                       float gamma, float lr) {
        Tensor<T, 2> input(1, 3);
        input(0, 0) = static_cast<T>(s.ball_x);
        input(0, 1) = static_cast<T>(s.ball_y);
        input(0, 2) = static_cast<T>(s.paddle_y);
        Tensor<T, 2> pred = model_->forward(input);
        Tensor<T, 2> input_next(1, 3);
        input_next(0, 0) = static_cast<T>(s_next.ball_x);
        input_next(0, 1) = static_cast<T>(s_next.ball_y);
        input_next(0, 2) = static_cast<T>(s_next.paddle_y);
        Tensor<T, 2> target_tensor = model_->forward(input_next);
        T target = static_cast<T>(reward) + static_cast<T>(gamma) * target_tensor(0, 0);
        Tensor<T, 2> y_true(1, 1);
        y_true(0, 0) = target;
        Tensor<T, 2> grad = pred - y_true;
        model_->backward(grad);
    }
    template class PongAgent<float>;
}