# asI-ADMM_RL
# Adaptive Stochastic ADMM for Decentralized Reinforcement Learning in Edge IoT

![GitHub last commit](https://img.shields.io/github/last-commit/alalulu8668/asI-ADMM_RL)
![GitHub contributors](https://img.shields.io/github/contributors/alalulu8668/asI-ADMM_RL)
![GitHub stars](https://img.shields.io/github/stars/alalulu8668/asI-ADMM_RL?style=social)

## Overview

This repository contains the code and resources related to the paper titled "Adaptive Stochastic ADMM for Decentralized Reinforcement Learning in Edge IoT." This paper presents a novel approach to decentralized reinforcement learning in the context of Edge IoT, leveraging Adaptive Stochastic Alternating Direction Method of Multipliers (ADMM) for efficient and scalable learning.

**Paper Link**: [Adaptive Stochastic ADMM for Decentralized Reinforcement Learning in Edge IoT](https://arxiv.org/abs/2107.00481)

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
Edge computing provides a promising paradigm to support the implementation of Internet of Things (IoT) by offloading tasks to nearby edge nodes. Meanwhile, the increasing network size makes it impractical for centralized data processing due to limited bandwidth, and consequently a decentralized learning scheme is preferable. Reinforcement learning (RL) has been widely investigated and shown to be a promising solution for decision-making and optimal control processes. For RL in a decentralized setup, edge nodes (agents) connected through a communication network aim to work collaboratively to find a policy to optimize the global reward as the sum of local rewards.  However, communication costs, scalability and adaptation in complex environments with heterogeneous agents may significantly limit the performance of decentralized RL. Alternating direction method of multipliers (ADMM) has a structure that allows for decentralized implementation, and has shown faster convergence than gradient descent based methods. Therefore, we propose an adaptive stochastic incremental ADMM (asI-ADMM) algorithm and apply the asI-ADMM to decentralized RL with edge-computing-empowered IoT networks. We provide convergence properties for proposed algorithms by designing a Lyapunov function and prove that the asI-ADMM has $\mathcal{O}(\frac{1}{k}) +\mathcal{O}(\frac{1}{M})$ convergence rate where $k$ and $ M$ are the number of iterations and batch samples, respectively. Then, we test our algorithm with two supervised learning problems. For performance evaluation, we simulate two applications in decentralized RL settings with homogeneous and heterogeneous agents. The experiment results show that our proposed algorithms outperform the state of the art in terms of communication costs and scalability, and can well adapt to complex IoT environments.


## Key Features

- The code contains adaptive stochastic ADMM algorithm. We evaluate the algorithm in decentralized least square regression and logistic regression. 
- Highlight the innovative or unique aspects of your approach.
- Include any performance benchmarks or results if available.

## Installation

Explain how to set up and install your project. Include detailed instructions for any dependencies and environment setup. For example:

