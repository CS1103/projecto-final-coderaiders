#include "utec/agent/EnvGym.h"
#include <cmath>

namespace utec::nn {
    EnvGym::EnvGym()
            : rng_(std::random_device{}()), dist_(0.0f, 1.0f)
    {
        reset();
    }
    State EnvGym::reset() {
        ball_x_    = dist_(rng_);
        ball_y_    = dist_(rng_);
        paddle_y_  = 0.5f;
        return {ball_x_, ball_y_, paddle_y_};
    }
    State EnvGym::step(int action, float& reward, bool& done) {
        if      (action < 0) paddle_y_ -= 0.05f;
        else if (action > 0) paddle_y_ += 0.05f;
        paddle_y_ = std::clamp(paddle_y_, 0.0f, 1.0f);
        if (std::fabs(ball_y_ - paddle_y_) < 0.1f) {
            reward = +1.0f;
            done   = false;
        } else {
            reward = -1.0f;
            done   = true;
        }
        ball_x_ = dist_(rng_);
        ball_y_ = dist_(rng_);
        return {ball_x_, ball_y_, paddle_y_};
    }
    State EnvGym::get_state() const {
        return {ball_x_, ball_y_, paddle_y_};
    }
}