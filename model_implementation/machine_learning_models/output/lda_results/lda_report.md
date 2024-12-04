# Linear Discriminant Analysis (LDA)

## Table of Contents

- [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Linear Discriminant Analysis for Classification Problems](#2-linear-discriminant-analysis-for-classification-problems)
    - [2.1 Basic Idea](#21-basic-idea)
    - [2.2. Objective Function in LDA](#22-objective-function-in-lda)
      - [2.2.1. Between-Class Scatter Matrix (( S\_B ))](#221-between-class-scatter-matrix--s_b-)
      - [2.2.2. Within-Class Scatter Matrix (( S\_W ))](#222-within-class-scatter-matrix--s_w-)
      - [2.2.3. Maximizing the Ratio of Between-Class and Within-Class Scatter](#223-maximizing-the-ratio-of-between-class-and-within-class-scatter)
  - [References](#references)
****

## 1. Introduction

Before going into the problem of LDA, I would like to recall the Principal component analysis (PCA) algorithm a bit. PCA is an unsupervised learning method, meaning it uses data vectors without considering class labels, if available. However, in classification tasks, where supervised learning is most common, using class labels can yield better classification results.

**PCA reduces the data dimensions to retain as much information (variance) as possible**. But in many cases, we don't need to preserve all the information—only the relevant data for the task at hand. For example, in a binary classification problem, data can be projected onto different lines, and the projection can impact how easily classes can be separated.

![alt text](https://machinelearningcoban.com/assets/29_lda/lda.png)

*Figure 1: Data projected onto different lines*

In the example (*Figure 1*), projecting the data onto different lines ($d_1$ and $d_2$) demonstrates that projecting along the first principal component ($d_1$) results in overlapping classes, making classification difficult. However, projecting along a secondary component ($d_2$) separates the classes clearly, facilitating better classification.

This suggests that preserving all the information from PCA does not always lead to the best results. The key is to find a projection that maximizes class separation. Linear Discriminant Analysis (LDA) addresses this issue by focusing on maximizing class separability in dimensionality reduction tasks.

**LDA is both a dimensionality reduction and a classification technique**. Unlike PCA, which is unsupervised, **LDA is a supervised method that seeks a projection that maximizes the separation between classes**. The new dimensionality will be at most $C-1$, where $C$ is the number of classes.

## 2. Linear Discriminant Analysis for Classification Problems

### 2.1 Basic Idea

LDA, like many classification methods, begins with binary classification. Returning to *Figure 1*, the bell curves represent the probability density functions (pdf) of the data projected onto different lines. The standard normal distribution is used as an approximation, though the data does not have to follow a normal distribution.

The spread of each curve reflects the standard deviation. A narrower spread indicates less dispersion, while a wider spread shows greater dispersion. Projecting data onto $d_1$ results in significant overlap between classes, while projecting onto $d_2$ leads to better separation, making classification more effective.

However, **small standard deviations alone do not guarantee good class separation**. *Figure 2* shows that the distance between class means (expectations) and the total variance impact the discriminability of the data. 

![alt text](https://machinelearningcoban.com/assets/29_lda/lda4.png)

*Figure 2: Class separation based on variance and distance between class means*

- In *Figure 2a)*, both classes have high variance, causing significant overlap.
- In *Figure 2b)*, the variance within each class is small, but the class means are too close, leading to overlap.
- In *Figure 2c)*, both the variance is small and the distance between class means is large, resulting in minimal overlap and better discrimination.

**In LDA, we aim to maximize the ratio of between-class variance to within-class variance.** "Within-class variance" reflects how similar the data points are within each class, such as $s^2_1$ and $s^2_2$, and "between-class variance" reflects how distinct the classes are from each other $(m_1 - m_2)^2$. LDA seeks to find a projection that maximizes this ratio to achieve the best possible separation between classes.

### 2.2. Objective Function in LDA

Linear Discriminant Analysis (LDA) aims to find a projection that maximizes the separation between classes while minimizing the variation within each class. This is done by finding a projection matrix \( W \) that maximizes the ratio of between-class variance to within-class variance. The objective function used in LDA is as follows:

\[
J(W) = \frac{W^T S_B W}{W^T S_W W}
\]

Where:
- \( W \) is the projection vector (or matrix) that maps the data into a lower-dimensional space.
- \( S_B \) is the between-class scatter matrix, which measures the variance between different class means and the overall mean of the data.
- \( S_W \) is the within-class scatter matrix, which measures the variance within each individual class.

#### 2.2.1. Between-Class Scatter Matrix (\( S_B \))

The **between-class scatter matrix** quantifies the dispersion of the class means from the overall mean of all the classes. In other words, it measures how far apart the different class centers (means) are from the overall mean of the dataset.

The formula for \( S_B \) is:

\[
S_B = \sum_{i=1}^{C} N_i (\mu_i - \mu)(\mu_i - \mu)^T
\]

Where:
- \( N_i \) is the number of samples in class \( i \).
- \( \mu_i \) is the mean of class \( i \).
- \( \mu \) is the overall mean of all classes.
- \( C \) is the total number of classes.

#### 2.2.2. Within-Class Scatter Matrix (\( S_W \))

The **within-class scatter matrix** measures the variance within each class. It is calculated by summing the covariance matrices for each class, which quantify how spread out the data points are within each class.

The formula for \( S_W \) is:

\[
S_W = \sum_{i=1}^{C} \sum_{x \in X_i} (x - \mu_i)(x - \mu_i)^T
\]

Where:
- \( X_i \) is the set of data points in class \( i \).
- \( \mu_i \) is the mean of class \( i \).
- \( x \) is a data point in class \( i \).

#### 2.2.3. Maximizing the Ratio of Between-Class and Within-Class Scatter

The main objective of Linear Discriminant Analysis (LDA) is to find a projection \( W \) that maximizes the separation between different classes while minimizing the variance within each class. This can be achieved by maximizing the ratio of the between-class scatter matrix \( S_B \) to the within-class scatter matrix \( S_W \). Specifically, the goal is to find the projection direction that makes the classes as distinct as possible, while keeping the spread of points within each class as small as possible.

To find the optimal projection \( W \), we solve the eigenvalue problem:

\[
S_W^{-1} S_B W = \lambda W
\]

Where:
- \( S_W^{-1} S_B \) is the matrix we need to solve for, which is a measure of how the between-class variance relates to the within-class variance after applying the inverse of the within-class scatter matrix.
- \( W \) represents the eigenvectors (directions of projection) corresponding to the largest eigenvalues, which indicate the most significant directions for maximizing class separability.
- \( \lambda \) represents the eigenvalue corresponding to each eigenvector, and the eigenvectors with the largest eigenvalues give the directions that maximize the ratio of between-class to within-class variance.

By solving this eigenvalue problem, we obtain the optimal projection matrix \( W \) that transforms the data into a lower-dimensional space where the classes are well-separated and the spread of points within each class is minimized.

After finding the optimal projection matrix \( W \), we can project the data onto this lower-dimensional space to perform classification tasks more effectively:

\[
Y = XW
\]

Where:
- \( Y \) is the transformed data in the lower-dimensional space.
- \( X \) is the original data.
- \( W \) is the optimal projection matrix.

## References

[1]. Vu, T. (n.d.). *Machine Learning cơ bản*. Tiep Vu’s Blog. https://machinelearningcoban.com/
[2]. Tharwat, A., Gaber, T., Ibrahim, A., & Hassanien, A. E. (2017). Linear discriminant analysis: A detailed tutorial. *AI Communications*, 30(2), 169–190. https://doi.org/10.3233/AIC-170729

