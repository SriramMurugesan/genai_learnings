# Probability Basics for AI

## Introduction

Probability theory is fundamental to understanding machine learning and AI. Most ML algorithms are based on probabilistic principles, from Bayesian inference to neural network training.

## Why Probability Matters in AI

- **Uncertainty**: Real-world data is noisy and uncertain
- **Predictions**: ML models output probabilities, not certainties
- **Learning**: Many algorithms optimize probabilistic objectives
- **Decision Making**: AI systems make decisions under uncertainty

## Basic Probability Concepts

### 1. Probability Fundamentals

**Definition**: Probability measures the likelihood of an event occurring.

**Properties**:
- 0 ≤ P(A) ≤ 1 for any event A
- P(certain event) = 1
- P(impossible event) = 0
- P(A or B) = P(A) + P(B) - P(A and B)

**Example**:
```
Rolling a die:
P(rolling a 3) = 1/6
P(rolling an even number) = 3/6 = 1/2
```

### 2. Random Variables

**Definition**: A variable whose value is determined by chance.

**Types**:
- **Discrete**: Countable outcomes (e.g., coin flips, dice rolls)
- **Continuous**: Infinite possible values (e.g., height, temperature)

### 3. Probability Distributions

#### Discrete Distributions

**Bernoulli Distribution**:
- Models binary outcomes (success/failure)
- Parameter: p (probability of success)
- Example: Coin flip, classification (yes/no)

**Binomial Distribution**:
- Number of successes in n independent Bernoulli trials
- Parameters: n (trials), p (success probability)
- Example: Number of heads in 10 coin flips

**Categorical Distribution**:
- Generalization of Bernoulli for multiple categories
- Example: Image classification (cat, dog, bird)

#### Continuous Distributions

**Normal (Gaussian) Distribution**:
- Most important distribution in statistics
- Parameters: μ (mean), σ² (variance)
- Bell-shaped curve
- Many natural phenomena follow this distribution

**Formula**: 
```
f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
```

**Properties**:
- Symmetric around mean
- 68% of data within 1 standard deviation
- 95% within 2 standard deviations
- 99.7% within 3 standard deviations

**Uniform Distribution**:
- All outcomes equally likely
- Example: Random number generation

### 4. Expected Value and Variance

**Expected Value (Mean)**:
- Average value of a random variable
- E[X] = Σ x * P(x) for discrete
- E[X] = ∫ x * f(x) dx for continuous

**Variance**:
- Measure of spread/dispersion
- Var(X) = E[(X - μ)²]
- Standard Deviation: σ = √Var(X)

### 5. Conditional Probability

**Definition**: Probability of A given that B has occurred

**Formula**: P(A|B) = P(A and B) / P(B)

**Example**:
```
P(rain | cloudy) = P(rain and cloudy) / P(cloudy)
```

**Independence**:
- Events A and B are independent if P(A|B) = P(A)
- Equivalently: P(A and B) = P(A) * P(B)

### 6. Bayes' Theorem

**Most important theorem in ML!**

**Formula**:
```
P(A|B) = P(B|A) * P(A) / P(B)
```

**In ML terms**:
```
P(hypothesis|data) = P(data|hypothesis) * P(hypothesis) / P(data)

Posterior = (Likelihood * Prior) / Evidence
```

**Applications**:
- Spam filtering
- Medical diagnosis
- Bayesian inference
- Naive Bayes classifier

**Example - Medical Test**:
```
Disease prevalence: P(Disease) = 0.01 (1%)
Test accuracy: P(Positive|Disease) = 0.99 (99%)
False positive rate: P(Positive|No Disease) = 0.05 (5%)

Question: If test is positive, what's probability of having disease?

P(Disease|Positive) = P(Positive|Disease) * P(Disease) / P(Positive)

P(Positive) = P(Positive|Disease)*P(Disease) + P(Positive|No Disease)*P(No Disease)
            = 0.99*0.01 + 0.05*0.99
            = 0.0594

P(Disease|Positive) = 0.99 * 0.01 / 0.0594 = 0.167 (16.7%)

Despite 99% accurate test, only 16.7% chance of having disease!
```

### 7. Joint and Marginal Probability

**Joint Probability**: P(A and B)
- Probability of both events occurring

**Marginal Probability**: P(A)
- Probability of A regardless of B
- P(A) = Σ P(A and B) for all B

### 8. Maximum Likelihood Estimation (MLE)

**Concept**: Find parameters that maximize probability of observed data

**Example - Coin Flip**:
```
Observed: 7 heads in 10 flips
What's the probability of heads (p)?

Likelihood: L(p) = C(10,7) * p^7 * (1-p)^3

MLE: p = 7/10 = 0.7
```

**Used in**:
- Training neural networks
- Fitting statistical models
- Parameter estimation

## Probability in Machine Learning

### 1. Classification

**Output**: Probability distribution over classes

```python
# Example: Image classification
P(cat|image) = 0.7
P(dog|image) = 0.2
P(bird|image) = 0.1
```

### 2. Loss Functions

**Cross-Entropy Loss**:
- Measures difference between predicted and true probability distributions
- Based on information theory
- Used in classification

### 3. Bayesian Machine Learning

**Idea**: Treat model parameters as random variables

**Benefits**:
- Uncertainty quantification
- Principled way to incorporate prior knowledge
- Avoid overfitting

### 4. Probabilistic Graphical Models

**Bayesian Networks**:
- Represent dependencies between variables
- Efficient inference

**Markov Models**:
- Sequential data modeling
- Hidden Markov Models (HMMs)

### 5. Sampling Methods

**Monte Carlo Methods**:
- Approximate complex distributions
- Used in Bayesian inference

**Markov Chain Monte Carlo (MCMC)**:
- Sample from complex distributions
- Gibbs sampling, Metropolis-Hastings

## Common Probability Distributions in ML

| Distribution | Use Case | Parameters |
|--------------|----------|------------|
| Bernoulli | Binary classification | p |
| Categorical | Multi-class classification | p₁, p₂, ..., pₖ |
| Gaussian | Continuous data, noise | μ, σ² |
| Poisson | Count data | λ |
| Exponential | Time between events | λ |
| Beta | Prior for probabilities | α, β |

## Key Formulas Summary

**Bayes' Theorem**:
```
P(A|B) = P(B|A) * P(A) / P(B)
```

**Law of Total Probability**:
```
P(A) = Σ P(A|Bᵢ) * P(Bᵢ)
```

**Expected Value**:
```
E[X] = Σ x * P(x)
```

**Variance**:
```
Var(X) = E[X²] - (E[X])²
```

## Practical Tips

1. **Always normalize**: Probabilities must sum to 1
2. **Log probabilities**: Prevent numerical underflow
3. **Conditional independence**: Simplifies computations
4. **Empirical distributions**: Estimate from data
5. **Visualization**: Plot distributions to understand them

## Common Pitfalls

1. **Confusing P(A|B) with P(B|A)** - Prosecutor's fallacy
2. **Ignoring base rates** - Base rate neglect
3. **Assuming independence** when variables are dependent
4. **Numerical instability** with very small probabilities

## Applications in Deep Learning

1. **Softmax Function**: Converts logits to probabilities
2. **Dropout**: Probabilistic regularization
3. **Variational Autoencoders**: Probabilistic latent variables
4. **Bayesian Neural Networks**: Uncertainty in weights
5. **Generative Models**: Learn probability distributions

## Summary

Probability is the language of uncertainty in AI:
- Quantifies uncertainty in predictions
- Enables principled decision-making
- Foundation for most ML algorithms
- Essential for understanding deep learning

**Key Takeaways**:
- Understand basic probability rules
- Master Bayes' theorem
- Know common distributions
- Apply to ML problems

---

**Next**: [Linear Algebra for AI](./02_linear_algebra.md)
