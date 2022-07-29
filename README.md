# Solving the Inverse Quantum Potential Problem Using Nelder-Mead and GPE
We use [Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) to find potentials that produce a state with desired expectation values admitting small standard deviations.

## Energy and Momentum in 1D
First, we seek minimum energy with momentum $k$.


# Todo
- [ ] tests
    - [ ] schrodinger.Schrodinger1D
    - [ ] nelder_mead.Objective
- [ ] schrodinger
    - [ ] add GPE term to get_dydt
- [ ] nelder_mead
  - [x] implement a distance from desired expectation value function
  - [x] try adaptive option in scipy minimize
  - [ ] explore space of init potentials
- [ ] Hyperparameters
  - [x] Select a method for Hyperparameter tuning
  - [x] implement method
