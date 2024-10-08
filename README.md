# Code for "Tuning Selection pressure..."

## Overview

[Publication](#publication)

[Brief description of Voter Model](#brief-description-of-voter-model)

[Structure of simulation code](#structure-of-simulation-code)

[Directory structure and exectuing code](#directory-structure-and-executing-code)

[Reproducing figures](#reproducing-figures)


## Publication

Manuscript published in XXX in 2024 by Thomas Tunstall. The corresponding preprint can be found on arXiv XXX.

We have included all data files and results pertinent to the manuscript (except those larger than 100MB).

## Brief description of voter model

The voter model takes place on a network (of $N$ nodes) where each node subscribes to an opinion, and at each simulation step a randomly selected node adopts the opinion of a random neighbour.

In this modified version, there are two opinions, a strong and weak version. The weak version has a fitness factor $F\leq1$ of being selected as a node updates its opinion. Furthermore, a proportion $z$ of the nodes are zealots, which subscribe to the weak opinion and never update their opinions.

## Structure of simulation code

### Network Creation

`GraphList`: A 1D numpy array of length $N$: each element holds an integer corresponding to the variety of node. 0 strong, 1 Weak, 2 Zealot.

`GraphNeighbourList`: A 1D numpy array of length $N$: each element holds a set of indices for neighbours of the node corresponding to the element.

`ChangeAbleList`: A 1D list of all nodes which have the ability to change in the next timestep.

`ActiveList`: A 1D list of length $N$: each element holds the number of active links a node possesses.

### Process

A random non-zealot is chosen to update its opinon: this is done by generating a random number i between 0 and N and ensuring that `GraphList[i]` isn't equal to 2.

If the chosen node is not in ChangeableList, then its opinion is unable to be updated by its neighbours as they all agree! Therefore we skip the rest of the simulation step, to save time.
On the other hand, if it is in ChangeableList, GraphNeighbourList is used to evaluate the proportion of neighbours who are strong. The probability of becoming/remaining strong is then determined: if this probability is met then it becomes/remains strong. Else, it becomes/remains weak.

If the node changed its opinion, the number of active links it has flips from its original value, A, to its new value L-A, where L is the number of neighbours the node has (given by `len(GraphNeighbourList)`). Then, if this number is $0$, the node is removed from the ChangeableList, as it now agrees with all of its neighbours.

If the node changed its opinion, each neighbour also also updates its number of active links in ActiveList - should the number drop to 0, it should similarly be removed from CHangeableList. Conversely, if the number of active links increases from 0, it should be added to ChangeableList. 

If we make the decision to try to simulate an infinite graph, to ensure the absorbing state is not met for the giant component we randomly select a weak node to become strong in the event that all the free nodes becomes weak.

## Directory structure and exectuing code

Below is a tree respresenting the structure of directories.

```
.
├── PaperFigs
├── RawFigures
└── Simulations
    ├── CoreFunctions
    ├── Fig1_InitialObservation
    ├── Fig2_CompleteDistComparisons
    ├── Fig2_CompleteDistComparisons_VaryN
    ├── Fig3_SmallC
    └── Fig4_FullTheory

```

### Directory `RawFigs`

This stores all the figures used in the publication. In order to populate it, execute:

```
$ bash Bash_Copying.sh RawFigs/
```

### Simulations

Here are the directories which house the code used to generate figures for the publication.

#### Fig1_InitialObservation

The case of a single $F,z$ choice to illustrate the nontrivial behviour of the system as $C$ is varied
![Initial Finding](./RawFigures/Fig1_c.png)

#### Fig2_CompleteDistComparisons 

For $P=0.1, \Phi=10^{-5}$ we vary the length of the Selection region and calculate the change in Resistant pests over a domain so large that the periodic effects of the circular convolution have negligible effect. The result is a detailed global minimum of change in Resistance number per Selection region size with changing Selection region size.

![Vary Params](./RawFigures/Fig2.png)


#### Fig2_CompleteDistComparisons_VaryN

For $P=0.1, \Phi=10^{-5}$ we run the same simulation as above, except with the additional option of restricting migration to only occur during the selection phase, the post-selection phase, both phases, or neither phase. Running once for each possibility with the same 

![Different N](./RawFigures/Fig2_b.png)


#### Fig3_SmallC

For $P=0.1, \Phi=10^{-5}$ we run an isolated selection region case for a given $L$, for the sake of visualising how the distribution of each subpopulation changes over a generation 

![Small C](./RawFigures/Fig3_c.png)

#### Fig4_FullTheory

We once again vary the size of the Selection region size $L$, but for different $P$ or $\Phi$ values in `Vary_PAP` and `Vary_Phi`, respectively. This is in order to measure how the region of highest curvature changes with these values, to validate the analytical theory.

![Full Approx Vary F](./RawFigures/Fig4_a.png)

![Full Approx Vary z](./RawFigures/Fig4_b.png)



## Reproducing figures

To recreate the data, navigate to the `Host Directory` and execute the corresponding commands.

| Figure(s) | Host Directory | Commands for simulation and creating figure |
| -------------| ------------- | ------------- |
| [1c](./RawFigures/Fig1_c.png) | `Simulations/Fig1_InitialObservation` |`cp SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_10.000_Cnum_41_NodeNum_10000_minZ_0.30000_maxZ_0.30000_Znum_1_minF_0.30000_maxF_0.30000_Fnum_1_Timesteps_1000000000_SingleActive_False_Repeats_2_LargestComponent_False_DataPoints_100000000/Params.py .` <br> `python FPRunFile.py` <br> `python MultiPlotting.py -d /SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_10.000_Cnum_41_NodeNum_10000_minZ_0.30000_maxZ_0.30000_Znum_1_minF_0.30000_maxF_0.30000_Fnum_1_Timesteps_1000000000_SingleActive_False_Repeats_2_LargestComponent_False_DataPoints_100000000` |
| [2a](./RawFigures/Fig2.png) |`Simulations/Fig2_CompleteDistComparisons` | `python Plotting.py` |
| [2b](./RawFigures/Fig2_b.png) |`Simulations/Fig2_CompleteDistComparisons_VaryN` | `python Plotting.py` |
| [3c](./RawFigures/Fig3_c.png) |`Simulations/Fig3_SmallC` | `cp SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_4.000_Cnum_21_NodeNum_10000_minZ_0.01000_maxZ_0.75000_Znum_3_minF_1.00000_maxF_1.00000_Fnum_1_Timesteps_1000000000_SingleActive_False_Repeats_100_LargestComponent_False_DataPoints_100000000/Params.py .` <br> `python FPRunFile.py` <br> `python MultiPlotting.py -d SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_4.000_Cnum_21_NodeNum_10000_minZ_0.01000_maxZ_0.75000_Znum_3_minF_1.00000_maxF_1.00000_Fnum_1_Timesteps_1000000000_SingleActive_False_Repeats_100_LargestComponent_False_DataPoints_100000000` |
| [4a](./RawFigures/Fig4_a.png) |`Simulations/Fig4_FullTheory/` | `cp SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_10.000_Cnum_41_NodeNum_10000_minZ_0.30000_maxZ_0.90000_Znum_3_minF_0.30000_maxF_0.30000_Fnum_1_Timesteps_1000000000_SingleActive_False_Repeats_10_LargestComponent_False_DataPoints_100000000/Params.py .` <br> `python FPRunFile.py` <br> `python MultiPlotting.py -d SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_10.000_Cnum_41_NodeNum_10000_minZ_0.30000_maxZ_0.90000_Znum_3_minF_0.30000_maxF_0.30000_Fnum_1_Timesteps_1000000000_SingleActive_False_Repeats_10_LargestComponent_False_DataPoints_100000000/` |
| [4b](./RawFigures/Fig4_a.png) |`Simulations/Fig4_FullTheory/` | `cp SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_10.000_Cnum_41_NodeNum_10000_minZ_0.30000_maxZ_0.30000_Znum_1_minF_0.30000_maxF_0.90000_Fnum_3_Timesteps_1000000000_SingleActive_False_Repeats_10_LargestComponent_False_DataPoints_100000000/Params.py .` <br> `python FPRunFile.py` <br> `python MultiPlotting.py -d SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_10.000_Cnum_41_NodeNum_10000_minZ_0.30000_maxZ_0.30000_Znum_1_minF_0.30000_maxF_0.90000_Fnum_3_Timesteps_1000000000_SingleActive_False_Repeats_10_LargestComponent_False_DataPoints_100000000/` |
