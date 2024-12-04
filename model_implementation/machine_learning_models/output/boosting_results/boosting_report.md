#  Boosting (LightBGM, XGBooost)
## Table of Contents 
- [Boosting (LightBGM, XGBooost)](#boosting-lightbgm-xgbooost)
  - [Table of Contents](#table-of-contents)
  - [Gradient Boosting](#gradient-boosting)
  - [XGBoost](#xgboost)
  - [LightGBM](#lightgbm)

Initially, the team planned to experiment with Random Forest, Ada Boosting, and Gradient Boosting. However, the first two algorithms do not currently have GPU-supported libraries, which makes feature selection and training of these algorithms on the massive EEG dataset take an extremely long time.

Therefore, the team decided to only experiment with the dataset on Gradient Boosting using the popular frameworks:
- LightGBM
- XGBoost
## Gradient Boosting
The _Gradient Boosting_ method shares the same idea as _AdaBoosting_, which is to train weak models sequentially. However, instead of using the model’s error to compute the weight for the training data, we use the residuals. Starting from the current model, we try to build a decision tree that attempts to match the residuals from the previous model. The special thing about this model is that instead of trying to match the target variable values, $y$, we try to match the error values of the previous model. Then, we add the trained model to the prediction function to gradually update the residuals. Each decision tree in the sequence has a very small size with only a few decision nodes, determined by the depth parameter $d$ in the model.

<center>
    <img src="https://i.imgur.com/YzvCJ6g.png" alt="Mô tả ảnh" width="400" height="400">
</center>

Assume that $\hat{f}(x)$ is the predicted function from the boosting method applied to the forecasting task with target variable $y$. At the $b$-th model in the forecast sequence - $\hat{f}^b$, we try to match the residuals $r_i$ from the previous decision tree $\hat{f}^{b-1}$. The steps of the algorithm are as follows:

1. Initially, set the prediction function $\hat{f}(\mathbf{x}) = 0$ and the residuals $\mathbf{r}_0 = \mathbf{y}$ for all observations in the training set.

2. Repeat the training process of decision trees sequentially with $b = 1, 2, \dots, B$. Each training round consists of the following sub-steps:

   a. Fit a decision tree $\hat{f}^{b}$ with depth $d$ on the training set $(\mathbf{X}, \mathbf{r}_b)$.

   b. Update $\hat{f}$ by adding the prediction of the decision tree, multiplied by the scaling factor $\lambda$:
   
   $$\hat{f}(\mathbf{x}) = \hat{f}(\mathbf{x}) + \lambda \hat{f}^{b}(\mathbf{x})$$

   c. Update the residuals for the model:

   $$\mathbf{r}_{b+1} := \mathbf{r}_b - \lambda \hat{f}^{b}(\mathbf{x})$$

   The algorithm stops updating when the number of decision trees reaches the maximum threshold $B$ or when all observations in the training set are predicted correctly.
URL của ảnh
3. The final prediction from the model sequence will be the combination of all sub-models:

   $$\hat{f}(\mathbf{x}) = \sum_{b=1}^{B} \lambda \hat{f}^{b}(\mathbf{x})$$

## XGBoost
XGBoost (Extreme Gradient Boosting) is an algorithm based on [[Gradient Boosting]], but with significant improvements in algorithm optimization, and a combination of software and hardware strength, which helps achieve exceptional results in both training time and resource usage.

XGBoost demonstrates remarkable capabilities:
- Solves regression, classification, ranking, and other user-defined problems effectively.
- High performance (fast training speed, memory optimization)
- Good overfitting prevention (regularization and shrinkage)
- Automatic handling of missing data
- High customization (parameters and loss functions)
- Support for parallel computation (CPU/GPU)
- Good model interpretability (feature importance)
- ...

<center>
    <img src="https://i.imgur.com/BqaYk2z.png" alt="Mô tả ảnh" width="600" height="400">
</center>

Since its first release in 2014, XGBoost has quickly gained popularity and is considered the main algorithm, producing outstanding results and winning top places in Kaggle competitions due to its simplicity and efficiency.

## LightGBM
Although XGBoost achieves outstanding results, it suffers from long training times, especially with large datasets. In January 2016, Microsoft released the experimental version of LightGBM, which quickly replaced XGBoost as the most popular ensemble algorithm.
<center>
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/20240308154358/LightGBM.webp" alt="Mô tả ảnh" width="600" height="400">
</center>

Key improvements of LightGBM over XGBoost include:
- LightGBM uses **histogram-based algorithms** instead of the **pre-sort-based algorithms** commonly used in other boosting tools to find the split point during tree construction. This helps LightGBM speed up training and reduce memory usage. A significant improvement of LightGBM over XGBoost is the inclusion of two algorithms:
  - GOSS (Gradient Based One Side Sampling)
  - EFB (Exclusive Feature Bundling)
  
  These algorithms significantly accelerate the computation process.
  
- LightGBM is based on **leaf-wise** growth, while most other boosting tools are based on **depth-wise** growth. Leaf-wise selects nodes to expand trees based on the overall optimization of the entire tree, while depth-wise only optimizes on the branch currently being considered. Therefore, with a smaller number of nodes, trees built from leaf-wise are generally more optimized than those built from depth-wise.

<center>
    <img src="https://files.codingninjas.in/article_images/lightgbm-0-1644216435.webp" alt="Mô tả ảnh" width="600" height="400">
</center>


One consideration when using LightGBM is that although leaf-wise is very effective, for smaller datasets, trees built with leaf-wise tend to overfit quickly. Therefore, LightGBM provides a hyperparameter `max_depth` to limit this. However, Microsoft recommends using LightGBM on sufficiently large datasets, which is the case for the EEG dataset in this problem.
