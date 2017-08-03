# quantum trajectories on UChicago midway

This repository contains all the code needed to execute massively parallel quantum trajectories on UChicago midway computating center.

The quantum trajectories technique involves rewriting the master equation as a stochastic average over individual trajectories, 
which can be evolved in time numerically as pure states. This avoids the need to propagate a full
density matrix in time, and replace this complexity with stochastic sampling. 

Quantum trajectories method is more computationally efficient for large system (Hilbert space >100), 
as it could directly leverage the massive parallelism provided by computing clusters.

The following shows the qubit **ground state** population under the action of a gaussian pulse.

This shows a single trajectory, where a quantum jump (excited state => ground state) happens stochastically in the middle of the pulse.

![Single trajectory](http://i.imgur.com/5Fl5XdW.png)

By averaging 100 individual trajectories, the final results matches quantitatively with the master equation.

![Average trajectory](http://i.imgur.com/yO5x5gY.png)

This code leverages SciPy's sparse matrix methods, allowing fast quantum simulation of Hilbert space dimension of over thousands.

For reference of quantum trajectories, refer to [this link](https://arxiv.org/abs/1405.6694).

## login to UChicago midway
`ssh **YOUR_CNetID**@midway1.rcc.uchicago.edu`

get [PuTTy](http://www.putty.org/) for ssh if you are using Windows.

## Clone this repository
`git clone https://github.com/SchusterLab/midway_quantum_trajectories.git`

## Run file
`cd batch`

`sbatch run.sbatch` for running 1 trajectory

`sbatch run_array.sbatch` for running 100 trajectories all at once. 

The run_array.sbatch script assigns 100 computation nodes to execute our program.

## Check run status
`squeue --user=**YOUR_CNetID**`

## Retrieve data
data stored at `/data/qubit`

