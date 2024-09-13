# **Feed Forward Neural Network (FFNN)**

This repository, **"1.2 Feed Forward Neural Network (FFNN)"**, is dedicated to understanding and implementing **Feed Forward Neural Networks (FFNN)** through a comprehensive learning strategy, supported by Google Colab notebooks. The concepts covered range from basic introductions to advanced optimization and techniques used in FFNNs, with additional sections on the **Motivation for Multi-Layer Perceptron** and **Miscellaneous Observations**.

---

## **Table of Contents**
1. [Introduction to FFNN](#introduction-to-ffnn)
2. [Impact of Activation Functions](#impact-of-activation-functions)
3. [Advantages of FFNN vs. Perceptrons](#advantages-of-ffnn-vs-perceptrons)
4. [Effect of Hidden Layers](#effect-of-hidden-layers)
5. [Challenges in Training Deeper FFNNs](#challenges-in-training-deeper-ffnns)
6. [Forward Propagation vs. Backpropagation](#forward-propagation-vs-backpropagation)
7. [Preventing Overfitting in FFNNs](#preventing-overfitting-in-ffnns)
8. [Data Normalization and Preprocessing](#data-normalization-and-preprocessing)
9. [Motivation for Multi-Layer Perceptron (MLP)](#motivation-for-multi-layer-perceptron-mlp)
10. [Miscellaneous Observations](#miscellaneous-observations)

---

## **Google Colab Integration**

For each section, the corresponding code has been implemented in a Google Colab notebook. These notebooks provide both the theory and practical implementation, using Python libraries like **TensorFlow**, **Keras**, and **scikit-learn**. The `.ipynb` files are included in the repository for interactive learning and experimentation.

---

## **1. Introduction to FFNN**
This section introduces the concept of **Feed Forward Neural Networks (FFNN)** and walks through the basic building blocks, including **input**, **hidden**, and **output layers**, along with forward propagation and activation functions like **ReLU** and **Sigmoid**.

- **Colab Notebook**: [Introduction to FFNN](./1.2_FFNN_Intro.ipynb)
- **Key Concepts**:
  - Understanding the structure of an FFNN.
  - Learning forward propagation through layers.
  - Exploring non-linear activation functions.

---

## **2. Impact of Activation Functions**
This section explores how the choice of activation functions (ReLU, Sigmoid) affects the performance of FFNNs. We compare the training speed, gradient behavior, and suitability of these functions in different tasks.

- **Colab Notebook**: [Impact of Activation Functions](./1.2_FFNN_Activation.ipynb)
- **Key Concepts**:
  - Performance comparison between **ReLU** and **Sigmoid**.
  - Understanding the impact on convergence and gradient flow.

---

## **3. Advantages of FFNN vs. Perceptrons**
This section compares **Feed Forward Neural Networks** with **Perceptrons** and discusses the advantages of using FFNNs for complex, non-linear problems.

- **Colab Notebook**: [FFNN vs. Perceptrons](./1.2_FFNN_vs_Perceptrons.ipynb)
- **Key Concepts**:
  - Limitations of single-layer perceptrons.
  - Solving non-linear problems with FFNNs.

---

## **4. Effect of Hidden Layers**
In this section, we explore the effect of increasing the number of hidden layers on an FFNN's learning capacity. The notebook demonstrates how deeper networks can capture more complex patterns.

- **Colab Notebook**: [Effect of Hidden Layers](./1.2_FFNN_Hidden_Layers.ipynb)
- **Key Concepts**:
  - The role of hidden layers in learning more abstract features.
  - How increasing hidden layers affects model capacity.

---

## **5. Challenges in Training Deeper FFNNs**
Here, we address the challenges that arise when training deeper FFNNs, such as the **vanishing gradient problem** and strategies like **batch normalization** and **ReLU** activation to mitigate these issues.

- **Colab Notebook**: [Challenges in Deep FFNN Training](./1.2_FFNN_Training_Challenges.ipynb)
- **Key Concepts**:
  - Understanding and solving the vanishing gradient problem.
  - Techniques for stabilizing and speeding up training in deep FFNNs.

---

## **6. Forward Propagation vs. Backpropagation**
This section explains the differences in computational complexity between **forward propagation** and **backpropagation**, while demonstrating how gradients are computed during backpropagation.

- **Colab Notebook**: [Forward vs. Backpropagation](./1.2_FFNN_Forward_Backpropagation.ipynb)
- **Key Concepts**:
  - How forward propagation calculates the output.
  - How backpropagation updates weights using the gradient of the loss function.

---

## **7. Preventing Overfitting in FFNNs**
In this section, we cover various strategies to prevent **overfitting** in FFNNs, including techniques like **dropout**, **L2 regularization**, and **early stopping**.

- **Colab Notebook**: [Preventing Overfitting](./1.2_FFNN_Overfitting_Prevention.ipynb)
- **Key Concepts**:
  - Regularization techniques to improve generalization.
  - Early stopping and cross-validation.

---

## **8. Data Normalization and Preprocessing**
This section demonstrates how **data normalization** impacts the training efficiency and performance of FFNNs. The notebook shows the difference between normalized and unnormalized data during training.

- **Colab Notebook**: [Data Normalization](./1.2_FFNN_Data_Normalization.ipynb)
- **Key Concepts**:
  - The importance of feature scaling.
  - How normalization speeds up convergence.

---

## **9. Motivation for Multi-Layer Perceptron (MLP)**
This section outlines the motivation for transitioning from single-layer models to **Multi-Layer Perceptrons (MLP)**. We explore how deeper networks solve non-linear problems and what makes them foundational to deep learning.

- **Colab Notebook**: [Motivation for MLP](./1.2_FFNN_MLP_Motivation.ipynb)
- **Key Concepts**:
  - Understanding the limitations of shallow models.
  - How MLPs leverage multiple layers to solve complex tasks.

---

## **10. Miscellaneous Observations**
In this final section, we compile various observations about FFNNs, such as the effect of insufficient hidden layers, how FFNNs compare to CNNs and RNNs, and best practices for model tuning.

- **Colab Notebook**: [Miscellaneous Observations](./1.2_FFNN_Observations.ipynb)
- **Key Concepts**:
  - What happens when you use too few hidden layers?
  - How do FFNNs stack up against CNNs and RNNs for different tasks?

---

## **Usage Instructions**

To use this repository:
1. Clone the repository to your local machine or open the links to the Google Colab `.ipynb` files.
2. Each section provides a mix of theory and practical coding exercises to demonstrate key concepts.
3. You can run the notebooks directly in Google Colab or modify them to experiment with different parameters.

---

## **How to Run in Google Colab**

1. Open any of the `.ipynb` files in this repository by clicking on the notebook links.
2. The notebooks will open in Google Colab, where you can run the code cells interactively.
3. Modify code, test different datasets, and observe results in real-time using Colab's powerful GPU/TPU resources.

---

## **License**

This repository is licensed under the MIT License. You are free to use, modify, and distribute the content in this repository, provided that proper attribution is given.

---

This README provides a structured learning path for **Feed Forward Neural Networks (FFNNs)**, including Colab notebooks for hands-on practice and theoretical understanding. Each section contains practical demonstrations to help learners grasp key concepts and optimize their neural network models efficiently.