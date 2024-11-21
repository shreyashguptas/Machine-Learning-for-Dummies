# Machine Learning for Dummies 🧠

Ever felt overwhelmed by Machine Learning jargon? We get it! This course explains Deep Learning concepts using simple analogies and practical code examples using PyTorch.

## Why This Course? 🤔

Learning Machine Learning and Deep Learning can be tough. While there are many great resources out there, we believe the best way to learn is through:
- Simple analogies that relate to everyday life
- Clear explanations of why we do what we do
- Hands-on coding examples
- No complicated math (just the essential concepts)

## What Makes This Course Different? 💡

- **Everything explained with analogies**: Complex topics broken down using real-world examples
- **Beginner-friendly**: No prior Machine Learning/Deep Learning knowledge needed
- **Learn by doing**: Practical code examples and exercises
- **AI-assisted learning**: Recommended use of AI chat tools (like Perplexity) when stuck

## Prerequisites 📚

- Basic Python programming knowledge
- Ability to run Python code (locally or using cloud platforms like Google Colab)
- Curiosity to learn!

## Course Content 📖 (Needs to be updated once the notebooks are finished)

### Machine Learning Fundamentals
- Linear Regression: Your First ML Model
- Gradient Descent: How Models Learn
- Normal Equations: A Different Approach

### Deep Learning Journey
- PyTorch Basics: Your New ML Friend
- Neural Networks: Building Blocks
- Convolutional Neural Networks (CNNs): Image Processing Magic
- Recurrent Neural Networks (RNNs): Understanding Sequences

## How to Use This Course 🎯

1. Go to the notebooks folder and open the notebook that you want to learn from 
2. Read every piece of text carefully and then run the code that you see below it oftentimes just write the code again yourself just type it out as you see above and run it yourself. 

## Getting Help 🆘

- **Stuck on a concept?** Use AI chat tools for personalized analogies
- **Found an issue?** Create a new issue in this repository
- **Need visual learning?** Check out [Amazon's MLU-Explain](https://mlu-explain.github.io/)

## Contributing 🤝

This is an open-source project! Feel free to:
- Suggest better analogies
- Add new examples
- Fix errors
- Share your learning experience

## Start Learning! 🚀

Ready to begin? Head to the first notebook in the `notebooks` folder!

---

Remember: Everyone learns differently. If this approach doesn't click with you, that's okay! Check out other resources like MLU-Explain for visual learning.



### Here is how to go about the course

[Introduction to Deep Learning with PyTorch.ipynb](notebooks/deep-learning/Introduction to Deep Learning with PyTorch.ipynb)

----

# Additional things to add. 
### 1. How do I know this is the right course for me to do? 
### (...)

### 2. Here is how this course is structured and intended to be used. 
This is important because if you go how it is intended you'll get the maximum amount of value out of this course.

## Installation 📦

There are two ways to use this repository:

1. **To run the code examples locally:**   ```bash
   pip install -r requirements.txt   ```
   This will install all the necessary ML/DL libraries needed to run the example code.

2. **To build the book locally:**   ```bash
   # Create and activate conda environment with book-building dependencies
   conda env create -f requirements-book.yml
   conda activate jupyter-book
   
   # Build the book
   jupyter-book build . --all`

The separation provides several benefits:
1. Faster deployments since only necessary packages are installed
2. Clearer separation of concerns between book building and code running
3. Conda environment for book building helps avoid dependency conflicts
4. Users can choose to install only what they need

You might also want to add a `.gitignore` entry for the conda environment: