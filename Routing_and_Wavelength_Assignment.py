
"""




Random Graph Generation and RWA ILP Optimization and Heuristic method for solving RWA Problem
Authors: Rosario Ietro, Alessandra Leo, Peyman Pahlevanzadeh
Date: June 2024


In this script we have two Heuristic methods for solving RWA Problem and 
If activate_gurobi = 0 the code of GUROBI will be activated. 



This script generates a random directed graph and traffic requests for a network. 
It formulates and solves the Routing and Wavelength Assignment (RWA) problem using 
Integer Linear Programming (ILP) with Gurobi, incorporating a hop constraint to limit 
the maximum number of nodes each lightpath can traverse.

Parameters:
- nodes: Number of nodes in the graph.
- edges: Number of edges in the graph.
- num_wavelengths: Number of available wavelengths.
- bitrate_max: Maximum bitrate for traffic requests (in GHz).
- bandwidth: Bandwidth per channel (in GHz).
- hop_max: Maximum number of nodes a lightpath can traverse (calculated as 2/3 of num_nodes).

Outputs:
- Optimized wavelength assignments and routed paths.
- Visualizations of the network graph and histogram of wavelength usage.

Dependencies:
- NetworkX: For graph generation and manipulation.
- Gurobi: For ILP formulation and optimization.
- Matplotlib: For plotting and visualization.
- Numpy: For numerical operations.
"""

import time
import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt
import random
from gurobipy import Model, GRB, quicksum
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import defaultdict






#Flags
activate_gurobi = 1 #If 0 doesn't launch gurobi simulation
activate_histograms = 1 #If 0 doesn't output histograms and graph for the single simulation

#Seed
seed = 250 #Random seed for reproducibility
random.seed(seed)

#Simulation parameters
num_wavelengths = 50 #Number of wavelengths/channels used
bitrate_max = 200  # Gbps, maximum bit rate request between two nodes
bandwidth = 50  # GHz, bandwidth associated to each channel. Assumed to be equal for all the channels.

#Graph parameters
nodes = [4] #Number of nodes used in the graph. It can contain multiple elements. In any case, ensure to put the varicontent inside square brackets.
edges = [12] #Number of edges used in the graph. It can contain multiple elements. In any case, ensure to put the content inside square brackets.







def RWA(activate_gurobi, seed, num_nodes, num_edges, num_wavelengths, bitrate_max, bandwidth, hop_max, activate_histograms):
    

    
    def generate_traffic_requests(graph_nodes, n_requests, bitrate_max, seed):
        
        random.seed(seed)
        np.random.seed(seed)
    
        """
        Generate traffic requests as traffic matrices with random bitrates.
        """
        traffic_requests = []
        for _ in range(n_requests):
            traffic_matrix = np.zeros((graph_nodes, graph_nodes))
            for i in range(graph_nodes):
                for j in range(i + 1, graph_nodes):
                    # Generate random bitrate between nodes i and j
                    traffic_matrix[i][j] = random.randint(0, bitrate_max)
                    traffic_matrix[j][i] = random.randint(0, bitrate_max)
            traffic_requests.append(traffic_matrix)
            
        return traffic_requests
    
    
    
    
    n_requests = 1
    traffic_requests = generate_traffic_requests(num_nodes, n_requests, bitrate_max, seed)
    
    print('Traffic Matrix: ')
    print(traffic_requests)
    
    
    
    
    def min_channels_requests(traffic_requests, bandwidth):
        """
        Calculate the minimum number of channels required for each traffic request.
        """
        channels_requests = []
        for request in traffic_requests:
            # Divide the request by bandwidth and round up to get the number of channels
            channels_requests.append(np.ceil(request / bandwidth))
        return channels_requests
    
    def extract_requests(traffic_matrix):
        """
        Extract individual requests from a traffic matrix.
        """
        requests = []
        for i in range(traffic_matrix.shape[0]):
            for j in range(traffic_matrix.shape[1]):
                if traffic_matrix[i][j] > 0:
                    # Create requests for each unit of traffic
                    for _ in range(int(traffic_matrix[i][j])):
                        requests.append((i, j))
        return requests
    
    def generate_random_graph_and_requests(num_nodes, num_edges, bitrate_max, bandwidth, seed, complete):
    
        
        
        random.seed(seed)
        np.random.seed(seed)
    
    
        """
        Generate a random directed graph and traffic requests.
        """

        # Generate a random directed graph
        if complete == True or num_edges > num_nodes*(num_nodes - 1):
            num_edges = num_nodes*(num_nodes - 1)
            
            
        G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
    
        
       
    
        # Generate traffic requests
        traffic_requests = generate_traffic_requests(num_nodes, 1, bitrate_max, seed)
        
        # Calculate the minimum number of channels required for each request
        new_TM = min_channels_requests(traffic_requests, bandwidth)
        
        # Print the traffic matrix
        for channels in new_TM:
            print('\nRequests Matrix:')
            print(channels)
        
        # Extract individual requests from the traffic matrix
        extracted_requests = extract_requests(new_TM[0])
        return G, extracted_requests, new_TM
    
    def formulate_rwa_ilp(graph, requests, num_wavelengths, hop_max):
        """
        Formulate the RWA ILP problem with the hop constraint.
        """
        model = gp.Model()
    
        # Extract nodes and edges from the graph
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        num_requests = len(requests)
    
        # Decision variables
        y = model.addVars(num_requests, num_wavelengths, vtype=GRB.BINARY, name="y")  # Wavelength assignment
        x = model.addVars(num_requests, num_wavelengths, len(edges), vtype=GRB.BINARY, name="x")  # Link usage
        u = model.addVars(num_wavelengths, vtype=GRB.BINARY, name="u")  # Wavelength usage indicator
    
        # Each request must be assigned exactly one wavelength
        for k in range(num_requests):
            model.addConstr(quicksum(y[k, w] for w in range(num_wavelengths)) == 1, name=f"assign_wavelength_{k}")
    
        # Link usage constraints
        for k, (s_k, t_k) in enumerate(requests):
            for w in range(num_wavelengths):
                for e, edge in enumerate(edges):
                    model.addConstr(x[k, w, e] <= y[k, w], name=f"link_usage_{k}_{w}_{e}")
    
        # Flow conservation constraints
        for k, (s_k, t_k) in enumerate(requests):
            for w in range(num_wavelengths):
                for n in nodes:
                    inflow = quicksum(x[k, w, e] for e, edge in enumerate(edges) if edge[1] == n)
                    outflow = quicksum(x[k, w, e] for e, edge in enumerate(edges) if edge[0] == n)
                    if n == s_k:
                        model.addConstr(outflow - inflow == y[k, w], name=f"flow_source_{k}_{w}_{n}")
                    elif n == t_k:
                        model.addConstr(outflow - inflow == -y[k, w], name=f"flow_sink_{k}_{w}_{n}")
                    else:
                        model.addConstr(outflow - inflow == 0, name=f"flow_intermediate_{k}_{w}_{n}")
    
        # Link capacity constraints
        for w in range(num_wavelengths):
            for e, edge in enumerate(edges):
                model.addConstr(quicksum(x[k, w, e] for k in range(num_requests)) <= 1, name=f"capacity_{w}_{e}")
    
        # Wavelength usage constraints
        for w in range(num_wavelengths):
            model.addConstr(quicksum(y[k, w] for k in range(num_requests)) <= num_requests * u[w], name=f"wavelength_usage_{w}")
    
        # Hop constraint: limit the maximum number of nodes traversed by each lightpath
        for k, (s_k, t_k) in enumerate(requests):
            for w in range(num_wavelengths):
                model.addConstr(quicksum(x[k, w, e] for e in range(len(edges))) <= hop_max, name=f"hop_constraint_{k}_{w}")
    
        # Objective: minimize the total number of wavelengths used
        model.setObjective(quicksum(u[w] for w in range(num_wavelengths)), GRB.MINIMIZE)
    
        return model
    
    
    def convert_digraph_to_multigraph(digraph):
    
        multigraph = nx.MultiDiGraph()
        
        multigraph.add_nodes_from(digraph.nodes(data=True))
        
        for u, v, data in digraph.edges(data=True):
            multigraph.add_edge(u, v, **data)
        
        return multigraph
    
    
    
    
    random.seed(seed)
    np.random.seed(seed)
    
    
    
    # Generate the random graph and requests
    graph, requests, traffic_matrix_bitrate  = generate_random_graph_and_requests(num_nodes, num_edges, bitrate_max, bandwidth, seed, complete=False)

    
    
    if activate_gurobi == 1:
        # Print the generated requests
        
        print(requests)
      
        
        # Record the start time
        start_time = time.time()
        
        # Formulate and optimize the ILP model
        model = formulate_rwa_ilp(graph, requests, num_wavelengths, hop_max)
        model.optimize()
        
        wavelengths_gurobi = model.ObjVal
        # Print optimization results and solving time
        print("\nOptimization Results:")
        print(f"Solving Time: {model.Runtime} seconds")
        print(f"Number of Variables: {model.NumVars}")
        print(f"Number of Constraints: {model.NumConstrs}")
        print(f"Objective Value: {model.ObjVal}")
        
        
        
        # Record the end time
        end_time = time.time()
        
        # Check and display the results
        # Check if the optimization model found an optimal solution
        if model.status == GRB.OPTIMAL:
            print("\nGurobi Results:")
            wavelength_usage = [0] * num_wavelengths
        
            # Loop over each request
            for k, (src, dst) in enumerate(requests):
                # Loop over each wavelength to find the assigned wavelength for the current request
                for w in range(num_wavelengths):
                    # Check if request k is assigned to wavelength w
                    # model.getVarByName(f"y[{k},{w}]").x retrieves the value of the variable y[k, w] in the solution
                    # The condition > 0.5 is used to handle numerical precision, interpreting the variable as 1 (true) if its value is greater than 0.5
                    if model.getVarByName(f"y[{k},{w}]").x > 0.5:
                        wavelength_usage[w] += 1
                        print(f"Request {k+1}: Source Node = {src}, Destination Node = {dst}, Routed Path:")
                        print(f"  Using Wavelength {w}")
                        
                        # Loop over each edge to find the edges used by the current request on the assigned wavelength
                        for e, edge in enumerate(graph.edges):
                            # Check if edge e is used by request k on wavelength w
                            # model.getVarByName(f"x[{k},{w},{e}]").x retrieves the value of the variable x[k, w, e] in the solution
                            # The condition > 0.5 is used to handle numerical precision, interpreting the variable as 1 (true) if its value is greater than 0.5
                            if model.getVarByName(f"x[{k},{w},{e}]").x > 0.5:
                                print(f"    Link {edge} is used")
                        break  # Exit the loop once the assigned wavelength is found
        
        
            # Sort wavelength usage in descending order and get the sorted indices
            sorted_indices = sorted(range(len(wavelength_usage)), key=lambda k: wavelength_usage[k], reverse=True)
            sorted_wavelength_usage = [wavelength_usage[i] for i in sorted_indices]
            
            if activate_histograms != 0:
                # Plotting the sorted histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(len(sorted_wavelength_usage)), sorted_wavelength_usage, color='red')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlabel('Wavelength', fontsize=14, fontweight='bold')
                ax.set_ylabel('Number of Requests', fontsize=14, fontweight='bold')
                ax.set_title('Histogram of Wavelength Usage (Gurobi)', fontsize=16, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.7)
        
                # Optionally, you can set y-axis limits to better visualize the histogram
                ax.set_ylim(0, max(sorted_wavelength_usage) + 1)
        
            # Calculate the execution time
            execution_time = end_time - start_time
        
            # Print the execution time
            print(f"Execution time: {execution_time} seconds")
    
            
        
        
           
        else:
            print("No optimal solution found.")
            pos = nx.kamada_kawai_layout(graph)
            nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, font_weight='bold', arrows=True)
            plt.title("Random Directed Graph")
            plt.show()
    
    
    
    
    
    
    
    def update_weights(G):
        '''
        Updates graph weights of edges after channel occupation
        '''
        for u, v, key, data in G.edges(data=True, keys=True):
            data['weight'] = np.sum(data['channels'])
            
        return G
    
    
    
    
    def satisfy_traffic_request(G, traffic_matrix, max_hops):
        '''
        Given the set of requests (traffic matrix), satisfies them all with minimum channel occupation
        (heuristic approach)
        '''
        n = 0 
        paths = []    
        rejected_connections = 0
        for j in range(len(traffic_matrix)):
            for k in range(len(traffic_matrix)):
                
                if j != k:
                    
                    
                    for i in range(int(traffic_matrix[j-1][k-1])):
                        n += 1
                        shortest_paths_length = available_paths_hops(G, j, k, max_hops) #Set of path with minimum number of nodes crossed (respecting the hop constraint)
                        shortest_path_channels = max_weight_path(G, shortest_paths_length) #Extract the path with maximum weight (that minimizes channels occupation)
                        available_lightpaths = np.prod(extract_channels(G, shortest_path_channels), axis=0) #Vector of 0s and 1s with length num_wavelengths: cells with 0 indicate lightpath occupied, while cells with 1 indicate free lightpath
                        if np.sum(available_lightpaths) != 0:
                            lightpath = np.min(np.nonzero(available_lightpaths)[0]) #Selects the lightpath with minimum index
                            G = occupy_channel(G, shortest_path_channels, lightpath) #Used channels are occupied
                            G = update_weights(G) #Weights updating
                            paths.append(shortest_path_channels)
                            print('Request', n, ': Source Node =', j, ', Destination Node =', k, 'Routed Path: \n Using wavelength', lightpath, '\n Path:', shortest_path_channels[0] )
                        else:
                            rejected_connections += 1
                            
        return G, paths, rejected_connections
        
                            
              
    
    
    def max_weight_path(graph, shortest_paths):
        '''
        Given a set of shortest paths, it finds the one with less channels occupied 
        (weight maximization)
        '''
        try:
            
            if not shortest_paths:
                return None, None
            max_weight = float('-inf')
            best_path = None
            best_edge_indices = None
            for path in shortest_paths:
                total_weight = 0
                edge_indices = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    best_edge_index = None
                    max_edge_weight = float('-inf')
                    for edge_index, edge_data in graph[u][v].items():
                        weight = edge_data['weight']
                        if weight > max_edge_weight:
                            max_edge_weight = weight
                            best_edge_index = edge_index
                    if best_edge_index is not None:
                        edge_indices.append(best_edge_index)
                        total_weight += max_edge_weight
                if total_weight > max_weight:
                    max_weight = total_weight
                    best_path = path
                    best_edge_indices = edge_indices
            return best_path, best_edge_indices
        except nx.NetworkXNoPath:
            return None, None
    
    
    
    
    
                    
            
            
    def extract_channels(G, path):
        '''
        It returns the channels vectors of each pair of sequential nodes in the path 
        '''
        nodes, edge_indices = path
        nodes_array = np.array(nodes)
        edge_indices_array = np.array(edge_indices)
        
        uv_keys = np.array([(nodes_array[i], nodes_array[i+1], edge_indices_array[i]) for i in range(len(nodes_array) - 1)])
        
        channels = [G[u][v][key]['channels'] for u, v, key in uv_keys]
        
        return np.array(channels)    
    
    
    def occupy_channel(G, path, lightpath):
        '''
        Occupies all the channels involved for a request satisfaction (one channel per pair of nodes in
                                                                       the path)
        '''
        nodes, edge_indices = path
        for i in range(len(nodes) - 1):
            G[nodes[i]][nodes[i+1]][edge_indices[i]]['channels'][lightpath] = 0
            
        return G
    
    
    def check_channels_used(G):
        '''
        Returns all informations on channels occupations for final statistics
        '''
        
        edge_data_arrays = []
        for u, v, key, data in G.edges(data=True, keys=True):
            if 'channels' in data:
                channels_array = np.array(data['channels'])
                edge_data_arrays.append(channels_array)
        edge_data_arrays = np.array(edge_data_arrays)        
        n_channels_used = edge_data_arrays.shape[1] - np.sum(np.prod(edge_data_arrays, axis=0))        
        
        return n_channels_used, edge_data_arrays, channels_array
    
    
    def available_paths_hops(G, source, target, hops):
        '''
        Returns a set of paths with minimum number of nodes, without exceed the hop constrain.
        '''
        available_paths = list(nx.all_shortest_paths(instance.G, source=source, target=target))
        for i in available_paths:
            if len(i) > hops:
                available_paths.remove(i)
                
        if len(available_paths) == 0: #If no one of the paths found respects the hop constraint, 
                                        #then set of available paths is recomputed instead of returning
                                        #an empty set
                                                      
            available_paths = list(nx.all_shortest_paths(instance.G, source=source, target=target))
        return available_paths
                
    
    # Record the start time
    
    
    def first_fit_rwa(graph, requests, num_wavelengths, hop_max):
        """Implements the First Fit heuristic for the RWA problem."""
        wavelength_usage = defaultdict(lambda: [0] * num_wavelengths)
        solution = {}
    
        for req_id, (src, dst) in enumerate(requests):
            try:
                path = nx.shortest_path(graph, src, dst)
            except nx.NetworkXNoPath:
                continue
            
            if len(path) - 1 > hop_max:
                continue
            
            assigned_wavelength = -1
            for w in range(num_wavelengths):
                if all(wavelength_usage[e][w] == 0 for e in zip(path[:-1], path[1:])):
                    for e in zip(path[:-1], path[1:]):
                        wavelength_usage[e][w] = 1
                    assigned_wavelength = w
                    solution[(req_id, w)] = path
                    break
    
            if assigned_wavelength == -1:
                print(f"Request {req_id + 1}: No available wavelength for path {path}")
    
        used_wavelengths = sum(any(w) for w in zip(*wavelength_usage.values()))
        if used_wavelengths > num_wavelengths:
            print("The problem is not feasible with the given number of wavelengths.")
            return None, None
        else:
            return solution, wavelength_usage
    
    
    # # Record the start time
    start_time_F = time.time()
    
    # Run the first fit heuristic
    best_solution, best_wavelength_usage = first_fit_rwa(graph, requests, num_wavelengths, hop_max)
    
    # Record the end time
    first_fit_time = time.time() - start_time_F
    # Print optimization results
    if best_solution is not None:
        print("\nFirst Fit Method Results:")
        wavelengths_first_fit = sum(any(w) for w in zip(*best_wavelength_usage.values()))
        #print(f"Number of Wavelengths Used: {wavelengths_first_fit}")
        #print(f"Best Solution: {best_solution}")
    
        for (req_id, w), path in best_solution.items():
            print(f"Request {req_id + 1}: Source Node = {requests[req_id][0]}, Destination Node = {requests[req_id][1]}, Routed Path:")
            print(f"  Using Wavelength {w}")
            print(f"  Path: {path}")
    
        if activate_histograms != 0:
            # Visualize the graph
            # Create a new figure for the graph plot
            fig, ax = plt.subplots(figsize=(8, 6))
        
            # Generate the layout for the graph
            pos = nx.kamada_kawai_layout(graph)
        
            # Plot the graph
            nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size=500, font_size=10, font_weight='bold', arrows=True, ax=ax)
        
            # Set the title
            ax.set_title("Random Directed Graph", fontsize=16, fontweight='bold')
        
            # Plot histogram of wavelength usage
            ax = plt.figure(figsize=(10, 6)).gca()
            wavelength_count = [sum(best_wavelength_usage[e][w] for e in best_wavelength_usage) for w in range(num_wavelengths)]
            bars = plt.bar(range(num_wavelengths), wavelength_count, color='green')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Wavelength', fontsize=14, fontweight='bold')
            plt.ylabel('Number of Requests', fontsize=14, fontweight='bold')
            plt.title('Histogram of Wavelength Usage (First Fit)', fontsize=16, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.7)
            #print(f"Number of Wavelengths Used First Fit: {sum(any(w) for w in zip(*best_wavelength_usage.values()))}")
        

       
    else:
        print("No solution found.")
        pos = nx.kamada_kawai_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, font_weight='bold', arrows=True)
        plt.title("Random Directed Graph")
        plt.show()
    
    
    start_time = time.time()
    
    class Instance(object):
        def __init__(self, G, num_wavelengths, bandwidth, requests, max_hops, channels_requests=None):
            self.G = G
            self.num_wavelengths = num_wavelengths
            self.bandwidth = bandwidth
            self.requests = requests
            self.channels_requests = channels_requests
            self.max_hops = max_hops
    
            
        def min_channels_requests(self):
            self.channels_requests = []
            for i in range(len(self.requests)):
                self.channels_requests.append(np.ceil(self.requests[i]/self.bandwidth))
            
            return self.channels_requests
        
        
        def set_channels(self):
            for u, v, key in self.G.edges(keys=True):
                self.G[u][v][key]['channels'] = [1] * self.num_wavelengths
    
    
            
    instance = Instance(convert_digraph_to_multigraph(graph), None, None, None, None)
    instance.bandwidth = bandwidth
    instance.num_wavelengths = num_wavelengths
    instance.channels_requests = traffic_matrix_bitrate
    instance.set_channels()
    instance.max_hops = hop_max
    update_weights(instance.G)
    
    #Solving RWA with heuristics
    print("\n\n\nHeuristics Results:")
    instance.G, paths, rejected_connections = satisfy_traffic_request(instance.G, instance.channels_requests[0], instance.max_hops)
    
    heuristics_time = time.time() - start_time
    
    #Statistics collection
    wavelengths_heuristics, all_edges_channels, channels_array = check_channels_used(instance.G)
    channel_used_for_each_edge = num_wavelengths - np.sum(all_edges_channels, axis=1)
    num_nodes_crossed = []
    for i in range(len(paths)):
        num_nodes_crossed.append(len(paths[i][0]))
    
    
    
    if activate_histograms != 0:
        plt.figure(figsize=(10, 6)).gca()
        
        plt.bar(list(range(0, instance.num_wavelengths)), len(instance.G.edges())-np.sum(all_edges_channels,axis=0), edgecolor='black')
        
        plt.title('Histogram of Wavelength Usage (Heuristics)', fontsize=16, fontweight='bold')
        plt.xlabel('Wavelength')
        plt.ylabel('Number of Requests', fontsize=14, fontweight='bold')
        plt.grid('true')
        
        plt.show()
    
    print('\n')
    print("****************************")
    print("****************************")
    print("****************************")
    print("****** FINAL RESULTS *******")
    print("****************************")
    print("****************************")
    print("****************************")
    
    
    # Print the length of best_solution.items()
    print(f"Total number of requests: {len(best_solution.items())}")
    
    # Print the maximum value of req_id
    #print(f"Maximum value of req_id: {max_req_id + 1}")  # Adding 1 to make it 1-based index for consistency
    
    
    
    print(f"Number of Wavelengths Used First Fit: {sum(any(w) for w in zip(*best_wavelength_usage.values()))}", 'out of ', num_wavelengths)
    print("Number of Wavelengths Used by Heuristic:", wavelengths_heuristics, 'out of ', num_wavelengths)
        
    if activate_gurobi != 0:
        print(f"Number of Wavelengths Used by Gurobi: {model.ObjVal}", 'out of ', num_wavelengths)
        print(f"Number of Variables(Gurobi): {model.NumVars}")
        print(f"Gurobi time: {model.Runtime} seconds")

    else:
        wavelengths_gurobi = None
    
    
    print('First fit time: ', first_fit_time)    
    print('Heuristics time: ', heuristics_time)
    

    
    
    
    
    
    return wavelengths_heuristics, wavelengths_first_fit, wavelengths_gurobi




edges = np.sort(edges)
heuristics = []
first_fit = []
gurobi_ = []
    
for num_nodes in nodes:
    heuristics_single_run = []
    first_fit_single_run = []
    gurobi_single_run = []
    hop_max = int(num_nodes * 2 / 3)
    
    for num_edges in edges:
        
        
        random.seed(seed)
        np.random.seed(seed)
        
        wavelengths_heuristics, wavelengths_first_fit, wavelengths_gurobi = RWA(activate_gurobi, seed, num_nodes, num_edges, num_wavelengths, bitrate_max, bandwidth, hop_max, activate_histograms)
        heuristics_single_run.append(wavelengths_heuristics)
        first_fit_single_run.append(wavelengths_first_fit)
        gurobi_single_run.append(wavelengths_gurobi)
        
    heuristics.append(heuristics_single_run)
    first_fit.append(first_fit_single_run) 
    gurobi_.append(gurobi_single_run)
    
    
    plt.plot(edges, heuristics_single_run, marker='s', linestyle='-', color='b', label='Heuristics', linewidth=1, alpha=0.7)
    plt.plot(edges, first_fit_single_run, marker='^', linestyle='-', color='g', label='First Fit', linewidth=1, alpha=0.7)
    plt.plot(edges, gurobi_single_run, marker='o', linestyle='-', color='r', label='Gurobi', linewidth=1, alpha=0.7)
    

    plt.xlabel('Edges')
    plt.ylabel('Channels Used')
    plt.title(f'Number of Channels Used with {num_nodes} Nodes')
    
    plt.grid('true')
    
    plt.legend()
    
    
    plt.show()


