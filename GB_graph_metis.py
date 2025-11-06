import numpy as np
import networkx as nx
from collections import deque
import pymetis

class GB_prop:
    def __init__(self, init_GB_num=2):
        self.init_GB_num = init_GB_num

    def calculate_quality(self, graph):
        if len(graph) <= 1:
            return 0.0
        return graph.number_of_edges() / len(graph)

    def split_ball(self, graph, split_GB_list):
        """Split granular-balls based on the graph structure"""
        if len(graph) == 1:
            split_GB_list.append(graph)
            return
        
        # The two nodes with the highest degree of selection are taken as the centers
        degrees = dict(graph.degree())
        if len(degrees) < 2:  # No splitting occurs when the number of nodes is less than two
            split_GB_list.append(graph)
            return
            
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:2]
        center_nodes = top_nodes[:2]
        
        # Allocate nodes to two centers
        clusters = self.assign_nodes_to_centers(graph, center_nodes)
        cluster1, cluster2 = clusters[center_nodes[0]], clusters[center_nodes[1]]
        
        graph1 = graph.subgraph(cluster1)
        graph2 = graph.subgraph(cluster2)
        
        if len(graph1) == 0 or len(graph2) == 0:
            split_GB_list.append(graph)
            return
        
        # Calculate qualities of granular-balls before and after splitting
        quality_before = self.calculate_quality(graph)
        quality1 = self.calculate_quality(graph1)
        quality2 = self.calculate_quality(graph2)
        quality_after = (quality1 + quality2) / 2.0
        
        if quality_before < quality_after:
            self.split_ball(graph1, split_GB_list)
            self.split_ball(graph2, split_GB_list)
        else:
            split_GB_list.append(graph)

    def assign_nodes_to_centers(self, G, centers):
        """Assign nodes to the nearest center (based on graph distance)"""
        center_nodes_dict = {center: set([center]) for center in centers}
        visited = set(centers)
        queues = {center: deque([center]) for center in centers}
        
        while any(queues.values()):
            for center in centers:
                if queues[center]:
                    current = queues[center].popleft()
                    for neighbor in G.neighbors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            center_nodes_dict[center].add(neighbor)
                            queues[center].append(neighbor)
        return center_nodes_dict

    def get_GB_graph(self, nx_graph):
        """Perform granular-ball coarsening and return the granular-ball coarsening graph"""
        if len(nx_graph) < 2:  # Do not coarsen the small graph
            return nx_graph, [list(nx_graph.nodes())]
        
        # Calculate the square root of N as the initial number of granular-balls
        n = len(nx_graph)
        init_GB_num = int(np.sqrt(n))

        # print(f"Generate {init_GB_num} initial granular-balls using pymetis")
        
        # Prepare data for pymetis: Create a temporary node mapping (original node → continuous index)
        nodes = list(nx_graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}
        
        # Build the adjacency list required for pymetis (using temporary indexes)
        adj_list = []
        for node in nodes:
            neighbors = [node_to_idx[neigh] for neigh in nx_graph.neighbors(node)]
            adj_list.append(neighbors)
        
        # Use pymetis for graph segmentation
        try:
            # Divide it into init_GB_num parts
            _, partitions = pymetis.part_graph(init_GB_num, adj_list)
        except Exception as e:
            print(f"pymetis segmentation failed: {e}, use the alternative plan")
            # Alternative plan: Use the node with the highest degree as the center
            degree_dict = nx_graph.degree()
            init_centers = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:init_GB_num]
            init_centers = [node for node, _ in init_centers]
            init_clusters = self.assign_nodes_to_centers(nx_graph, init_centers)
            init_partitions = {node: i for i, (center, nodes) in enumerate(init_clusters.items()) for node in nodes}
            partitions = [init_partitions[node] for node in nodes]
        
        # Map the segmentation result back to the original node
        init_clusters = {}
        for idx, part_id in enumerate(partitions):
            node = idx_to_node[idx]
            if part_id not in init_clusters:
                init_clusters[part_id] = []
            init_clusters[part_id].append(node)
        
        # Filter empty clusters
        init_clusters = {k: v for k, v in init_clusters.items() if v}
        
        # Split granular-balls
        GB_list = []
        for cluster_nodes in init_clusters.values():
            subgraph = nx_graph.subgraph(cluster_nodes)
            self.split_ball(subgraph, GB_list)
        
        # Create coarsened graph
        GB_graph = nx.Graph()
        for i, cluster in enumerate(GB_list):
            GB_graph.add_node(i, nodes=list(cluster.nodes()))
            
        # Add coarsened edge
        for i in range(len(GB_list)):
            for j in range(i+1, len(GB_list)):
                # Optimize edge counting: Use set intersection
                nodes_i = set(GB_list[i].nodes())
                nodes_j = set(GB_list[j].nodes())
                edge_count = 0
                # Traverse smaller sets of nodes to improve efficiency
                if len(nodes_i) > len(nodes_j):
                    nodes_i, nodes_j = nodes_j, nodes_i
                for u in nodes_i:
                    for v in nx_graph.neighbors(u):
                        if v in nodes_j:
                            edge_count += 1
                            break  # Just find one to avoid double counting
                if edge_count > 0:
                    GB_graph.add_edge(i, j, weight=edge_count)
        
        return GB_graph, [list(cluster.nodes()) for cluster in GB_list]

    def transform_features(self, original_features, clusters):
        """Calculate the coarsened features (intra-cluster average value)"""
        coarse_features = []
        for cluster in clusters:
            if not cluster:  # Avoid empty clusters
                continue
            cluster_features = original_features[cluster]
            coarse_features.append(np.mean(cluster_features, axis=0))
        return np.array(coarse_features)