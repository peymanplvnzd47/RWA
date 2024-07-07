# Simulation Instructions
Random Graph Generation and RWA ILP Optimization and Heuristic method for solving RWA Problem
Authors: Rosario, Ale, Peyman 
Date: June 2024


In this script, we have two Heuristic methods for solving RWA Problem and 
If activate_gurobi = 0 the code of GUROBI will be activated. 



This script generates a random directed graph and traffic requests for a network. 
It formulates and solves the Routing and Wavelength Assignment (RWA) problem using 
Integer Linear Programming (ILP) with Gurobi, incorporates a hop constraint to limit 
the maximum number of nodes each lightpath can traverse.
This document provides detailed instructions for setting the input parameters for the simulation. Please follow the guidelines below to ensure the correct configuration.

## Input Parameters

### Flags

- **activate_gurobi**:
  - **Type**: Integer (0 or non-zero)
  - **Description**: Controls whether Gurobi simulations are run. If set to `0`, only heuristics and first-fit simulations will be run, which is useful when the Gurobi solver requires too much time to find a solution.

- **activate_histograms**:
  - **Type**: Integer (0 or non-zero)
  - **Description**: Controls the display of histograms for channel distribution and graph topology for a single run. If set to `0`, histograms will not be displayed, which is useful when we are interested in observing output behavior only across different simulations.

### Seed

- **seed**:
  - **Type**: Integer
  - **Description**: Random seed to reproduce the same simulation. In all experiments, it is set to `250`.

### Simulation Parameters

- **num_wavelengths**:
  - **Type**: Integer
  - **Description**: Number of channels/wavelengths used in the simulation.

- **bitrate_max**:
  - **Type**: Integer (Gbps)
  - **Description**: Maximum bit rate that can be requested from a source node to a destination node.

- **bandwidth**:
  - **Type**: Integer (GHz)
  - **Description**: Bandwidth occupied by each channel, assumed to be equal for all channels. In all experiments, it is set to `50 GHz`.

### Graph Parameters

- **nodes**:
  - **Type**: List of integers
  - **Description**: Number of nodes in the graph used in simulations. This can contain more than one element, so multiple simulations will be run. Ensure this variable is a list, even if it contains just one element (e.g., `nodes = [7]`).

- **edges**:
  - **Type**: List of integers
  - **Description**: Number of edges in the graph used in simulations. This can contain more than one element, so multiple simulations will be run. Ensure this variable is a list, even if it contains just one element (e.g., `edges = [20]`).

## Simulation Execution



For a single number of nodes and a single number of edges, a graph is determined, and a simulation is run. If `activate_histograms` is not `0`, histograms for channel distribution using the heuristic, first-fit method, and (if `activate_gurobi` is not `0`) Gurobi will be displayed. Regardless, text statistics will be printed at the end of the simulation.

### Example Output

Total number of requests: 27
Number of Wavelengths Used First Fit: 4 out of 50
Number of Wavelengths Used by Heuristic: 4 out of 50
Number of Wavelengths Used by Gurobi: 4.0 out of 50
Number of Variables (Gurobi): 17600
Gurobi time: 47.41000008583069 seconds
First fit time: 0.001001119613647461 seconds
Heuristics time: 0.009999275207519531 seconds

