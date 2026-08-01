# LBTree: A Learned Index for Conjunctive Keyword Queries

This repository provides the implementation of **LBTree**, a learned bitmap-tree index for **conjunctive keyword queries**.

Unlike existing keyword-query indexes that primarily optimize index structures based on data distributions, LBTree jointly learns from both **data distributions** and **historical query workload distributions** to improve **query efficiency** while controlling **storage overhead**.

The framework consists of two stages:

1. **LBTree-B Construction:** A binary learned tree is recursively constructed using **cost-aware binary bitmap clustering**.
2. **LBTree Optimization:** A tree-based dynamic programming algorithm selectively breaks internal nodes in LBTree-B, producing the final **adaptive-fanout LBTree** with lower query and storage costs.

## Features

- **LBTree-B**: Binary learned bitmap tree constructed by recursive binary partitioning
- **LBTree**: Final adaptive-fanout index obtained by breaking redundant internal nodes
- **Workload-aware indexing**: Jointly models object distributions and historical query workloads
- **Cost-guided construction**: Balances query latency and storage overhead with a unified cost model
- **Streaming assignment**: Supports scalable clustering on large datasets
- **Update support**: Supports object insertion, deletion, and local subtree rebuilding under data or workload drift

---

## Repository Structure

```text
LBTree/
├── baseline_compare/      # Baseline-related scripts or comparisons
├── index/                 # C++ query engine and runtime
├── model/                 # Python code for clustering, cost modeling, and tree construction
├── utils/                 # Utility modules
├── CMakeLists.txt         # CMake build script for the C++ executable
├── parameter.py           # Global parameters and checkpoint paths
├── test.py                # Main Python entry point for tree construction
└── README.md
```

---

## Requirements

This project uses **Python** for model training and tree construction, and **C++** for query-time evaluation.

### Python

Recommended version:

- Python 3.9+

Suggested packages:

```bash
pip install numpy pandas scipy torch openpyxl
```

### C++

Recommended environment:

- CMake 3.12+
- A C++17 compiler
- Linux environment recommended

---

## Overview

LBTree is designed for **pure keyword queries**, where objects do not have a natural order and existing learned indexing ideas cannot be directly applied.

The framework consists of two main stages:

1. **Binary Bitmap Clustering**
   - Builds the binary tree **LBTree-B**
   - Uses a cost-aware clustering model to partition objects
   - Incorporates both keyword overlap and workload information

2. **Node Breakage by Dynamic Programming**
   - Optimizes the binary tree structure
   - Breaks redundant internal nodes
   - Produces the final adaptive-fanout **LBTree**

---

## Input Data

LBTree assumes that:

- Each **object** is associated with a set of keywords
- Each **query** is also represented as a keyword set
- Historical query logs are used as the **workload** for index construction

You need to prepare:

- Object-keyword data
- Query workload data
- Vocabulary or token mapping if required by the runtime

---

## Configuration

Before running the code, please check and modify the dataset paths and output paths in the following files according to your local environment:

- `parameter.py`
- `test.py`
- `index/src/main.cpp`
- `CMakeLists.txt`

The current code may contain machine-specific absolute paths, so manual adjustment may be necessary before reproduction.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/CHAN-dong/LBTree.git
cd LBTree
```

### 2. Configure local paths

Update the paths in the following files to match your local environment:

- `parameter.py`
- `test.py`
- `index/src/main.cpp`
- `CMakeLists.txt`

### 3. Build the tree in Python

Run the main Python script for binary tree construction and adaptive-fanout optimization:

```bash
python test.py
```

This step will typically:

- Load the object and workload data
- Construct **LBTree-B**
- Break redundant internal nodes to produce **LBTree**
- Write the resulting tree structure to output directories

### 4. Compile the C++ query engine

```bash
mkdir -p build
cd build
cmake ..
make -j
```

### 5. Run query evaluation

After compilation, run the executable:

```bash
./lbtree
```

This step will typically load the generated tree files, evaluate query performance, and report storage and latency statistics.

---

## Main Components

### Cost Model

LBTree uses a unified cost model to balance:

- **Query cost**
- **Storage cost**

This cost model guides both clustering and final tree optimization.

### Binary Bitmap Clustering

The binary clustering stage partitions the object set into two subsets while considering both:

- Data similarity
- Query workload similarity

It uses a learned clustering model together with a cost-aware objective.

### Streaming Assignment

To scale to large datasets, LBTree first learns clustering patterns from sampled objects and then incrementally assigns remaining objects by comparing assignment costs.

### Adaptive Fanout Optimization

After constructing the binary tree, LBTree applies a tree-based dynamic programming procedure to break redundant internal nodes and reduce overall cost.

## Experimental Datasets

The experimental evaluation in the associated paper uses four datasets, including three real-world datasets and one synthetic dataset:

| Dataset | Description |
|---------|-------------|
| **MARCO** | A widely used real-world passage retrieval benchmark based on the **MS MARCO Passage Ranking** dataset. It contains **8,841,823** passage objects and **1,010,916** queries. In the experiments, **1,000** queries are randomly sampled for index performance evaluation, while a randomly selected **10% of the remaining queries** are used as the workload for index construction. |
| **SciDocs** | A real-world academic retrieval dataset from the **BEIR** benchmark for scientific document retrieval. Each object corresponds to a paper entry, and keywords are extracted from textual fields such as the title and abstract, as well as metadata including authors, publication year, and citations. It contains **25,657** objects and **1,000** queries, of which **900** are used for index construction and the remaining **100** for evaluation. |
| **DBLP** | A real-world bibliographic dataset constructed from the official **DBLP** source. It contains **7,772,904** papers. Each paper is treated as an object, and every word in its title is extracted as a keyword. The workload is generated from ten groups of popular keywords covering ten computer science areas, with **10,000** queries used for index construction and another **10,000** for evaluation. |
| **SYN** | A synthetic dataset containing **10,000,000** objects and **10,000** distinct keywords. Each object is associated with **100** keywords sampled uniformly at random. Each query contains **2–10** keywords sampled from a Gaussian distribution over the keyword vocabulary. The experiments use **1,000** queries for index construction and another **1,000** for evaluation. |

---

## Notes

- This repository is research code intended for experimental evaluation
- Some scripts may require manual path adjustment before running
- The Python and C++ parts are designed to work together for construction and evaluation

---
