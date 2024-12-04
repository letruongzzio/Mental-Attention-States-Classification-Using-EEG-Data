# EEGNet

## 1. Theory of EEGNet

**EEGNet** is a specialized Convolutional Neural Network (CNN) designed specifically for processing EEG (Electroencephalography) data. It is characterized by:

- **Lightweight Architecture**: EEGNet is optimized for efficiency, particularly when working with EEG data, which has a high temporal resolution and typically a limited number of channels.
  
- **Key Architectural Components**:
  
  - **Input Shape**: The input data has the shape $(\text{Chans}, \text{Samples}, 1)$, where:
    - `Chans`: Number of EEG channels (electrodes).
    - `Samples`: Number of temporal samples per channel.
    - `1`: The single-channel input dimension.
  
  - **Block 1**:
    - **Temporal Feature Extraction**: A $\text{Conv2D}$ layer with filters $F_1$ and kernel length $\text{kernLength}$ is used to extract temporal patterns from each channel. The output shape is $(\text{Chans}, \text{Samples}, F_1)$.
    - **Spatial Feature Extraction**: A $\text{DepthwiseConv2D}$ layer captures inter-channel relationships by applying depthwise convolution. A depth multiplier $D$ is used to control the number of spatial filters. Output shape: $(1, \text{Samples}, F_1 \times D)$.
    - **Normalization**: Batch Normalization stabilizes and accelerates the learning process. The formula for normalization is:
      
      $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
      
      Where:
      - $\mu$: Batch mean.
      - $\sigma^2$: Batch variance.
      - $\epsilon$: Small constant for numerical stability.
    
    - **Scaling and Shifting**: The normalized values are scaled and shifted using:
      
      $$y_i = \gamma \hat{x}_i + \beta$$
      
      Where $\gamma$ and $\beta$ are learnable parameters.
    
    - **Activation (ELU)**: The Exponential Linear Unit (ELU) introduces non-linearity, improving learning for inputs near 0. The formula is:
      
      $$f(x) =
      \begin{cases}
      x, & \text{if } x > 0 \\
      \alpha (e^x - 1), & \text{if } x \leq 0
      \end{cases}$$
      
      Where $\alpha$ (typically 1) controls saturation for negative values.
    
    - **Pooling**: Reduces spatial dimensions by averaging data points. Output shape: $(1, \text{Samples}/4, F_1 \times D)$.
    - **Dropout**: Randomly removes some elements to prevent overfitting. In EEGNet, the default $\text{dropoutRate} = 0.5$.
    
    - **Output of Block 1**: The down-sampled data has the shape $(1, \text{Samples}/4, F_1 \times D)$.
  
  - **Block 2**:
    - **Separable Convolution**: A $\text{SeparableConv2D}$ layer integrates both spatial and temporal features from **Block 1** by using separable convolutions. This refines the learned representations while maintaining computational efficiency.
    - **Normalization, Activation, Pooling, and Dropout**: These steps help stabilize learning and enhance feature selection for classification.
    - **Output of Block 2**: The data is further down-sampled, with the shape $(1, \text{Samples}/16, F_2)$.

  - **Final Layers**:
    - **Flatten**: Converts the data into a one-dimensional vector. Output shape: $(\text{Samples}/16 \times F_2)$.
    - **Dense**: Aggregates the features and computes logits for the output classes. Each output class $j$ is computed as:
      
      $$y_j = \sum_{i=1}^n z_i W_{ij} + b_j$$
      
      Where:
      - $z_i$: Input features from the Flatten layer.
      - $W_{ij}$: Trainable weights connecting $z_i$ to $y_j$.
      - $b_j$: Bias term for class $j$.
      - $y_j$: Logits for class $j$ before applying Softmax.
      - $n = \text{Samples}/16 \times F_2$: Size of the input vector.

    The **norm rate** is enforced on the weight matrix $W$ to ensure:
      
      $$||W_j||_2 \leq 0.25$$
      
      (In EEGNet, $\text{norm rate} = 0.25$ by default).
    
    - **Softmax**: Converts logits into probabilities for classification. The formula for softmax is:
      
      $$P(y_j) = \frac{e^{y_j}}{\sum_{k=1}^{nb\_classes} e^{y_k}}$$
      
      Where:
      - $P(y_j)$: Probability of class $j$.
      - $nb\_classes = 3$: Number of classification categories (Focused, Unfocused, Drowsy).
      - The sum of probabilities for all classes is always 1.

  - **Final Output**: Probabilities for each class, with shape $(nb\_classes)$.

## 2. Parameters

- `nb_classes`: Number of output classes to classify.
- `Chans`, `Samples`: Number of EEG channels and temporal samples.
- `kernLength`: Length of temporal convolution in the first layer.
- `F_1`, `F_2`: Number of temporal and pointwise filters to learn (Default: $F_1 = 8$, $F_2 = F_1 \times D$).
- `D`: Number of spatial filters to learn within each temporal convolution (Default: $D = 2$).
- `norm_rate`: Limitation on kernel weights in the Dense layer to avoid overfitting.
- `dropoutRate`: Dropout fraction.
- `dropoutType`: Randomly removes neurons to prevent overfitting.

## 3. Training

- **Optimizer**: Adam with an adaptive learning rate for efficient convergence.
- **Loss**: Categorical Cross-Entropy for multi-class classification.
- **Batch Size**: 32, Epochs: 50.
- **Validation Set**: Used to monitor and improve model performance during training.

## 4. Optimization for EEG Data

The use of **Depthwise and Separable Convolutions** ensures a balance between computational efficiency and the ability to learn spatiotemporal features from EEG data.

## 5. Reasons for Choosing EEGNet for EEG Data in This Task

- **Tailored for EEG Data**:
    - The compact architecture processes EEG data efficiently, capturing both temporal and spatial patterns crucial for classifying psychological states.

- **Suitability for EEG Data**:
    - EEG data typically features a small number of channels but large timepoints. EEGNet is designed to process such data using specialized convolutional operations.
    - For classifying the three states (focused, unfocused, drowsy), EEGNet can learn both spatial features (across channels) and temporal features (patterns over time).

- **Good Generalization**:
    - EEGNet has been tested on various EEG datasets, showing its ability to generalize well for EEG-based classification tasks.

- **Ease of Customization**:
    - Parameters like `F_1`, `F_2`, `D`, and `kernLength` can be tuned to match dataset characteristics, making it suitable for tasks like classifying focused, unfocused, and drowsy states.

- **Computational Efficiency**:
    - EEGNet’s compact design saves computational resources and reduces training time.

- **Relevance to the Task**:
    - Psychological states are often represented by changes in EEG frequency bands. EEGNet, which captures both frequency and spatial information, is optimal for this task.
