#pragma once
#include <random>
#include <algorithm>

namespace utec::nn {
    struct State {
        float ball_x;
        float ball_y;
        float paddle_y;
    };
    class EnvGym {
    private:
        float ball_x_;
        float ball_y_;
        float paddle_y_;
        std::mt19937 rng_;
        std::uniform_real_distribution<float> dist_;
    public:
        EnvGym();
        State reset();
        State step(int action, float& reward, bool& done);
        State get_state() const;
    };
}