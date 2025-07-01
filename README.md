# 🔍 Nearest Neighbor Feature Selection

## Overview

This interactive Python program performs **feature selection** using a 1-Nearest Neighbor (1-NN) classifier with **Leave-One-Out Cross Validation (LOOCV)**. It's designed to evaluate which features in a dataset contribute most to classification accuracy.

Developed by **Fatima & Arya** for cs172 project.

---

## Features

- Supports custom datasets with class labels and numeric features.
- Implements three feature selection algorithms:
  - **Forward Selection** – Greedily adds best features.
  - **Backward Elimination** – Removes least impactful features.
  - **Arya's Special Algorithm** – Selects top 5 individual features based on performance.
- Tracks and prints:
  - Subsets and their accuracies
  - Final selected features
  - Training time

---

## Algorithms

### Forward Selection
Starts with an empty set and adds features that improve classification accuracy the most at each step.

### Backward Elimination
Starts with all features and removes the one that harms accuracy the least.

### Arya's Special Algorithm
Ranks all features by evaluating them individually and selects the **top 5** based on accuracy.

---

## Example Results

### Small Dataset
- Forward: `{3, 5}` → **92%**
- Backward: `{2, 4, 5, 7, 10}` → **83%**

### Large Dataset
- Forward: `{27, 1}` → **99.5%**
- Backward: `{27}` → **84.7%**

### Titanic Dataset
- Both Forward & Backward: `{2}` → **78%**

---

## Getting Started

### 📦 Prerequisites

- Python 3.x
- NumPy

Install required libraries:

```bash
pip install numpy
```

### Dataset Format
- .txt or .csv file with format
- First column = class labels & Remaining columns = numeric features
- example:
  ```
  1 0.6 0.2 0.9
  2 0.1 0.4 0.3
  1 0.8 0.5 0.7
  ```


## Running  Program
```sh
git clone https://github.com/arya-sth/NearestNeighbor.git
cd NearestNeighbor
python feature_selector.py
```

## Example Output 
```sh
Welcome to Fatima & Arya's Feature Selection Algorithm.

Type the name of the file to test: small-test-dataset.txt

Type the number of the algorithm you want to run.
1) Forward Selection
2) Backward Elimination
3) Arya's Special Algorithm.

> ...

Evaluating subset [3], Accuracy: 78.33%
Evaluating subset [5], Accuracy: 83.33%
Adding feature 5 to the final set

...

Final selected features: [3, 5], Max Accuracy: 92.00%
Time to train: 3.42 seconds
```

