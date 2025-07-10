[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

Pong AI es un proyecto académico desarrollado en C++ que tiene como objetivo crear un agente inteligente capaz de aprender a jugar el clásico videojuego Pong a través del entrenamiento con una red neuronal multicapa. A diferencia de soluciones que dependen de librerías externas, este proyecto implementa desde cero:

 * Una biblioteca algebraica genérica (Tensor<T, Rank>) similar a NumPy, que permite operar con tensores multidimensionales.

 * Un framework de redes neuronales minimalista, incluyendo capas densas, funciones de activación, cálculo de pérdida y optimización.

 * Un entorno de simulación simplificado (EnvGym) que modela las físicas básicas del juego Pong.

 * Un agente de decisión (PongAgent) que utiliza la red neuronal entrenada para actuar y mejorar en tiempo real.

Este proyecto no solo busca demostrar el aprendizaje por refuerzo en un entorno sencillo, sino también reforzar habilidades de diseño modular, patrones de software y programación genérica en C++20.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `Coderaiders`
* **Integrantes**:

  * Alumno A – Benalcázar Ferro, José Ignacio (Responsable de investigación teórica)
  * Alumno B – Luciani Dávila, Itzel Yadira Arellys (Desarrollo de la arquitectura)
  * Alumno C – Cervantes Ordóñez, Jireh Eliseo (Implementación del modelo y Pruebas)
  * Alumno D – Cabanillas Solis, Joseph Jossemy (Documentación y demo)



---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/ItzelLuci/projecto-final-coderaiders.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```


---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales a través del estudio de múltiples modelos y conceptos principales.
* **Contenido de ejemplo**:


  1. Historia y evolución de las NNs.

  
     * 1943: McCulloch y Pitts crean el primer modelo de neurona artificial.


     * 1958: Rosenblatt presenta el perceptrón, pero es limitado (no resuelve XOR).


     * 1986: Se redescubre el algoritmo de retropropagación, permitiendo entrenar redes multicapa.


     * 2010s: Surgen las redes profundas (deep learning), impulsadas por grandes datos y GPUs.


     * Hoy: Las NNs se aplican en visión, lenguaje, salud, juegos, etc.

  2. Principales arquitecturas: MLP, CNN, RNN.

    **MLP (Multilayer Perceptron)**:
    Redes con varias capas densas, útiles en tareas generales de clasificación y regresión.


    **CNN (Convolutional Neural Network)**:
    Extraen características espaciales, ideales para imágenes y visión por computadora.


    **RNN (Recurrent Neural Network)**:
    Procesan datos secuenciales como texto o series de tiempo. LSTM y GRU mejoran su memoria.

  3. Algoritmos de entrenamiento: backpropagation, optimizadores.


     **Backpropagation**:
     Calcula cómo ajustar los pesos para reducir el error, usando derivadas y la regla de la cadena.


    **Optimización**:
    Algoritmos que usan esos gradientes para mejorar la red.
     - SGD: Método básico
     - Adam: El más usado por su rapidez y adaptabilidad
     - Otros: Momentum, RMSprop

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas**:

```bash
pong_ai/
├── docs/                        # Documentación
│   ├── BIBLIOGRAFIA
│   └── README.md
├── include/
│   └── utec/
│       ├── agent/               # Agente Pong y entorno
│       │   ├── EnvGym.h
│       │   └── PongAgent.h
│       ├── algebra/            # Operaciones algebraicas
│       │   └── Tensor.h
│       └── nn/                 # Red neuronal
│           ├── activation.h
│           ├── dense.h
│           ├── layer.h
│           ├── loss.h
│           └── neural_network.h
├── src/                         # Implementación
│   └── utec/
│       └── agent/
│           ├── EnvGym.cpp
│           └── PongAgent.cpp
├── tests/                       # Casos de prueba
│   ├── test_agent_env.cpp
│   ├── test_neural_network.cpp
│   └── test_tensor.cpp
└── main.cpp                     # Punto de entrada
```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.



---

### 3. Ejecución

> **Demo de ejemplo**:  aqui pones el video del demo
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Alumno A | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno C | Generación de métricas    |
| Documentación y demo      | Alumno D | Tutorial y video demo     |

---

### 6. Conclusiones

El proceso de propagación hacia adelante (forward pass) y retropropagación (backward pass) permite ajustar los pesos y sesgos de la red para optimizar su rendimiento.

Dividir el proyecto en módulos independientes es importante para garantizar un código escalable y fácil de mantener. Sin embargo, podríamos extender el código mediante patrones como Dependency Injection para desacoplar más cada capa.

---

### 7. Bibliografía

- Blog, G. (2020, 5 julio). The Evolution and Core Concepts of Deep Learning & Neural Networks. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2016/08/evolution-core-concepts-deep-learning-neural-networks

   
- David, D. S. (2024). Aplicación de deep learning para predicción de índices bursátiles extranjeros usando modelos multivariados recurrentes y convolucionales con mecanismos de atención. repositorio.uchile.cl. https://doi.org/10.58011/nymx-3240

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
