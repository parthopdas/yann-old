# Yann - Yet Another Neural Network Library

## Purpose

- Reimplement material from [deeplearning.ai](https://www.deeplearning.ai/) in F# to gain deeper understanding.
- Build a .NET Core DNN library for other projects.

## Features

- Uses [Math.NET Numerics](https://numerics.mathdotnet.com/) [~33% slower than numpy implementation for Cats vs non-Cats Week 4 example on ThinkPad X1 Extreme]
- Customizable L-layer FC DNN Library (Unit tested + Demo apps)
- Activation: ReLU, Sigmoid
- Hyperparameters: L, α, Epochs
- Initialization: He
- Basic transfer learning

## Primary DNN Workflow

> Image (c) [@SkalskiP](https://github.com/SkalskiP)

![Deep Neural Network Workflow](./content/images/DNNWorkflow.gif "Deep Neural Network Workflow")

## References

- [deeplearning.ai: Deep Learning specialization](https://www.deeplearning.ai/) course.
- [ILearnDeepLearning.py](https://github.com/SkalskiP/ILearnDeepLearning.py/tree/master/01_mysteries_of_neural_networks/03_numpy_neural_net)
- [UFLDL Tutorial](http://ufldl.stanford.edu/tutorial/).
