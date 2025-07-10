cmake_minimum_required(VERSION 3.30)
project(pong_ai)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(pong_core
        src/utec/agent/EnvGym.cpp
        src/utec/agent/PongAgent.cpp
)

add_executable(test_tensor         tests/test_tensor.cpp)
add_executable(test_neural_network tests/test_neural_network.cpp)
add_executable(test_agent_env      tests/test_agent_env.cpp)
add_executable(main_exec           main.cpp)

target_link_libraries(test_tensor         pong_core)
target_link_libraries(test_neural_network pong_core)
target_link_libraries(test_agent_env      pong_core)
target_link_libraries(main_exec           pong_core)