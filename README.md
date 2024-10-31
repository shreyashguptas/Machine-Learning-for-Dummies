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

# Creators Note
Learning these concepts like Machine Learning and Deep Learning is hard. All the tehcnical jargon can get overwhelming. There are thousands of resources out there and all are good but each has a different way of teaching and you have to know what your best way of learning is. For us it is learning with Analogies. Only when the most complex topics are explained with analogies is when we learn and we believe that there are others who also want that way of learning. That is why we created this. Each topic will have an analogy to each. The smallest things will have an easy explanation on why we are doing what.
This repository will always be opensouce so if there is something that you didn't understand then feel free to create a new issue on the respository pointing out the exact issue. But this course is best learned paired with an AI Chat. We use Perplexity where when we get stuck we ask it questions. We ask it to teach us with analogies and we would recommend you do the same whenever you feel stuck. One of the best resources of our times.

## Introduction
This repository is all about how I started learning deep learning using PyTorch at the very basic level. It's very hard for me to understand concepts by just reading the definition. So this repository is going to be all about how I use analogies. I use AI to use analogies to learn certain topics and all of those topics analogies that I liked and helped me understand. I'm going to use those and put those in this repository so anyone can learn deep learning like a dummy. I am a dummy when it comes to learning these things and this repository has helped me learn and I've created this from scratch using the course from DataCamp called Introduction to Deep Learning with PyTorch and other sources as I move forward on building this repository.


As part of my deep learning journey, I will be following the Stanford CS229 Machine Learning course (Spring 2022). This renowned course, taught by Tengyu Ma and Chris Ré, provides a comprehensive introduction to machine learning concepts and techniques. Throughout this repository, I will be documenting my learnings, insights, and practical implementations based on the course material.

You can find the course lectures on YouTube:
[Stanford CS229: Machine Learning Course](https://www.youtube.com/playlist?list=PLoROMvodv4rNyWOpJg_Yh4NSqI4Z4vOYy)
[CS229 Lecture Notes](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf)

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
- [Linear Regression](./notebooks/supervised-learning/05_linear_regression.ipynb)
- [Batch & Stochastic Gradient Descent](./notebooks/supervised-learning/06_batch_and_stochastic_gradient_descent.ipynb)
- [Normal Equations](./notebooks/supervised-learning/07_normal_equation.ipynb)

### Deep Learning
- [Introduction to PyTorch](./notebooks/01_introduction_to_pytorch.ipynb)
- [Building Neural Networks](./notebooks/02_building_neural_networks.ipynb)
- [Convolutional Neural Networks (CNNs)](./notebooks/03_convolutional_neural_networks.ipynb)
- [Recurrent Neural Networks (RNNs)](./notebooks/04_recurrent_neural_networks.ipynb)

