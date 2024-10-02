# Learning 'deep-learning' for Dummies with pytorch library (Still cooking this repo)

## Table of Contents
- [Learning 'deep-learning' for Dummies with pytorch library (Still cooking this repo)](#learning-deep-learning-for-dummies-with-pytorch-library-still-cooking-this-repo)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
  - [Topics Covered](#topics-covered)
    - [Machine Learning](#machine-learning)
    - [Deep Learning](#deep-learning)

## Introduction
This repository is all about how I started learning deep learning using PyTorch at the very basic level. It's very hard for me to understand concepts by just reading the definition. So this repository is going to be all about how I use analogies. I use AI to use analogies to learn certain topics and all of those topics analogies that I liked and helped me understand. I'm going to use those and put those in this repository so anyone can learn deep learning like a dummy. I am a dummy when it comes to learning these things and this repository has helped me learn and I've created this from scratch using the course from DataCamp called Introduction to Deep Learning with PyTorch and other sources as I move forward on building this repository.


As part of my deep learning journey, I will be following the Stanford CS229 Machine Learning course (Spring 2022). This renowned course, taught by Tengyu Ma and Chris Ré, provides a comprehensive introduction to machine learning concepts and techniques. Throughout this repository, I will be documenting my learnings, insights, and practical implementations based on the course material.

You can find the course lectures on YouTube:
[Stanford CS229: Machine Learning Course](https://www.youtube.com/playlist?list=PLoROMvodv4rNyWOpJg_Yh4NSqI4Z4vOYy)

I'll be updating this repository with notes, code examples, and projects inspired by the course content, providing a practical, hands-on approach to understanding machine learning concepts.

## Getting Started

### Installation
To get the appropriate libraries and their versions, I have created a file called environment.yml. And what that's going to do is if you run the following commands below after you clone this repository to your local machine, it will install all of the required packages and libraries and their exact versions. So you'll never have conflicts of different versions of different libraries that don't work together.

```
conda env create -f environment.yml -p ./.conda
```

```
conda activate ./.conda
```

## Topics Covered


### Machine Learning
- [Linear Regression](./notebooks/05_linear_regression.ipynb)
- [Logistic Regression](./notebooks/06_logistic_regression.ipynb)
- [Decision Trees](./notebooks/07_decision_trees.ipynb)
- [Random Forests](./notebooks/08_random_forests.ipynb)
- [Support Vector Machines](./notebooks/09_support_vector_machines.ipynb)

### Deep Learning
- [Introduction to PyTorch](./notebooks/01_introduction_to_pytorch.ipynb)
- [Building Neural Networks](./notebooks/02_building_neural_networks.ipynb)
- [Convolutional Neural Networks (CNNs)](./notebooks/03_convolutional_neural_networks.ipynb)
- [Recurrent Neural Networks (RNNs)](./notebooks/04_recurrent_neural_networks.ipynb)

