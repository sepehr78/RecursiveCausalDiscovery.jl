
# RecursiveCausalDiscovery.jl
A Julia implementation of [Recursive Causal Discovery](https://arxiv.org/abs/2403.09300) algorithms. Recursive Causal Discovery (RCD) is an efficient approach for causal discovery (i.e., learning a causal graph from data).

#### ⚠️ This package is still under development! ⚠️
[![CI](https://github.com/sepehr78/RecursiveCausalDiscovery.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/sepehr78/RecursiveCausalDiscovery.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/sepehr78/RecursiveCausalDiscovery.jl/graph/badge.svg?token=ELQDCTLFCT)](https://codecov.io/github/sepehr78/RecursiveCausalDiscovery.jl)

# Overview
### Comparison of RCD with the PC algorithm implemented in [CausalInference.jl](https://github.com/mschauer/CausalInference.jl)
The following plots show the F1 score (computed using true causal graph), and number of conditional independency (CI) tests done.

<img src="rcd_vs_pc.png" alt="F1 score and #CI tests of RSL versus PC" width="500"/>

### Implemented algorithms
 - [x] Recursive Structure Learning (RSL)
   - [x] Learning v-structures
   - [x] Meek's rules
 - [ ] MArkov boundary-based Recursive Variable ELimination (MARVEL)
 - [ ] Latent MARVEL (L-MARVEL)
 - [ ] Removable Order Learning (ROL)
 

# Installation
Requires at least Julia 1.10
```julia
julia> ]add RecursiveCausalDiscovery
```

# How to use
The package so far has only one algorithm implemented: RSL-D, which can be called using the `rsld` function. The function takes the data matrix/table and a conditional independence test function as input, and returns the completed partially oriented directed acyclic graph (CPDAG) as output.

## Simple example
In this example, a csv file named `data.csv` is loaded, and the RSL-D algorithm is used to learn the CPDAG. The conditional independence test is based on the Fisher's Z-test.

```julia
using RecursiveCausalDiscovery
using CSV
using Tables

# load data (columns are variables and rows are samples)
data = CSV.read("data.csv", Tables.matrix)

# use a Gaussian conditional independence test
sig_level =  0.01
ci_test = (x, y, cond_vec, data) ->  fisher_z(x, y, cond_vec, data, sig_level)

# learn the oriented causal graph using RSL
cpdag =  rsld(data, ci_test)
```

See the [examples/rsl_example_wo_data.jl](examples/rsl_example_w_data.jl) for a complete example.

## Generating random data from DAG and learning from it
See the [examples/rsl_example_wo_data.jl](examples/rsl_example_wo_data.jl) for an example on how to generate a random DAG using the [Erdos-Renyi model](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model), generate random Gaussian data from it, and learning the CPDAG using RSL-D.

# Citation
If you do use this package, please cite our [work](https://arxiv.org/abs/2403.09300).

```bibtex
@misc{mokhtarian2024rcd,
      title={Recursive Causal Discovery}, 
      author={Ehsan Mokhtarian and Sepehr Elahi and Sina Akbari and Negar Kiyavash},
      year={2024},
      eprint={2403.09300},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.09300}, 
}
```