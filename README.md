# Counterfactual Regret Minimization Algorithms in Imperfect Information Games

This repository contains an implementation of several Counterfactual Regret Minimization (CFR) algorithms for solving imperfect information games, focusing on their performance in simplified poker variants. The project aims to provide a clear comparative analysis of the original CFR algorithm and its key modifications, examining their convergence rates and scalability in different game scenarios.

## Key Features

* **Multiple CFR Variants:** Implementation of Vanilla CFR, Discounted CFR (DCFR), CFR+, Outcome Sampling Monte Carlo CFR (OS-MCCFR), and CFR+ with Quadratic Weighted Averaging.
* **Benchmark Games:** Evaluation of algorithms in Leduc Hold'em and a newly introduced variant, Short Deck Big Leduc Hold'em (SDBLH), designed to assess scalability to more complex problems.
* **Exploitability Metric:** Assessment of algorithm performance based on exploitability, a measure of how much a strategy can be exploited by a theoretically optimal opponent.
* **Comparative Analysis:**  Detailed comparison of algorithm convergence speeds and relative performances, including analysis of the impact of alternating updates and different averaging schemes.
* **Scalability Analysis:**  Exploration of how algorithm performance scales with increasing game complexity, using SDBLH as a stepping stone towards larger games.
* **Clear Implementation:** Object-oriented design promoting code reuse and facilitating the addition of new CFR variants and game environments. 
* **OpenSpiel Integration:** Leverages OpenSpiel for game representation and utility functions, simplifying implementation and validation.
* **Detailed Documentation:** Comprehensive documentation outlining the theoretical framework, technical implementation, and experimental results.


## Usage

The main script (`main.py`) allows you to run different CFR variants with various parameters.  See the script and accompanying documentation for specific instructions on how to run experiments and generate results.

## Results

The project's findings provide valuable insights into the relative strengths and weaknesses of each CFR variant.  Key observations include the superior performance of DCFR with alternating updates, the diminishing returns of quadratic weighted averaging in CFR+ for complex games, and the limitations of comparing sampling-based methods (like OS-MCCFR) with deterministic algorithms based solely on iteration count.

## Future Work

Potential areas for future investigation include:

* **More Equitable Comparisons for Sampling Methods:** Employing metrics like "nodes touched" to better compare sampling-based methods with deterministic algorithms.
* **In-depth Performance Analysis:**  Exploring the specific reasons behind the observed performance differences in different game scenarios.
* **Extension to Larger Games:**  Applying the implemented algorithms to larger and more complex games to further analyze scalability.
* **Hybrid Approaches:**  Investigating the combination of CFR with other techniques, such as deep learning, to enhance performance.


## Contributing

Contributions are welcome!  Please feel free to submit issues or pull requests.
