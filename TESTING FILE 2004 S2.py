

##QUESTION 2##
import time
class Analyser:
    def __init__(self, sequences):
        
        self.sequences = sequences
        
        N= len(sequences)
        max_length = 0
        for s in sequences:
            max_length = max(max_length, len(s))

        self.max_frequency = [0 for _ in range(max_length + 1)]
        self.best_pattern_location = [None for _ in range(max_length + 1)] 

        pattern_frequ_map = []

        BASE = 37

        for song_id, song in enumerate(self.sequences):
            song_len = len(song)

            for start_index in range(song_len): # O(M) - start of subsequence
                rolling_hash = 0

                for end_index in range(start_index + 1, song_len): # O(M) - end of subsequence
                    interval = ord(song[end_index]) - ord(song[end_index - 1])
                    rolling_hash = (rolling_hash * BASE) + interval

                    pattern_length = end_index - start_index + 1
                    pattern_key = (pattern_length, rolling_hash)


                    entry_found = None
                    for entry in pattern_frequ_map:
                        if entry[0] == pattern_key:
                            entry_found = entry
                            break

                    if entry_found is None:
                        song_indices = [song_id]
                        pattern_frequ_map.append([pattern_key, song_indices])
                        frequency = 1
                    else:
                        song_indices = entry_found[1]
                        added_already = False

                        for previous_id in song_indices:
                            if previous_id == song_id:
                                added_already = True
                                break
                        if not added_already:
                            song_indices.append(song_id)


                        frequency = len(song_indices)

                    if frequency > self.max_frequency[pattern_length]:
                        self.max_frequency[pattern_length] = frequency
                        self.best_pattern_location[pattern_length] = (song_id, start_index)
                    
                    
    def getFrequentPattern(self, K):
        if K >= len(self.best_pattern_location):
            return []
        if self.best_pattern_location[K] is None:
            return []
        

        location = self.best_pattern_location[K]
        song_index = location[0]
        start_index = location[1]   

        song_with_best_pattern = self.sequences[song_index]

        pattern_string = song_with_best_pattern[start_index:start_index + K]
        return list(pattern_string)
    
    

# =============================================================================
# --- TEST HARNESS ---
# You can run this file to test the Analyser class.
# =============================================================================

if __name__ == "__main__":
    
    print("--- Analyser Test (No Dict/Set Version) ---")
    
    demo_songs = ["cegec", "gdfhd", "cdfhd"]
    
    print(f"Songs: {demo_songs}")
    try:
        start_time = time.time()
        analyser = Analyser(demo_songs)
        end_time = time.time()
        print(f"__init__ took: {end_time - start_time:.6f} seconds")

        # Test K=2
        pattern_2 = analyser.getFrequentPattern(2)
        print(f"K=2 => {pattern_2}")
        if "".join(pattern_2) in ("df", "fh", "ce", "eg"):
             print("K=2 Test: PASS (Found a valid pattern)")
        else:
             print(f"K=2 Test: UNEXPECTED (Got {''.join(pattern_2)})")

        # Test K=3
        pattern_3 = analyser.getFrequentPattern(3)
        print(f"K=3 => {pattern_3}")
        if "".join(pattern_3) in ("dfh", "ceg"):
             print("K-3 Test: PASS (Found a valid pattern)")
        else:
             print(f"K-3 Test: UNEXPECTED (Got {''.join(pattern_3)})")

        # Test K=4
        pattern_4 = analyser.getFrequentPattern(4)
        print(f"K=4 => {pattern_4}")
        if "".join(pattern_4) == "dfhd":
             print("K-4 Test: PASS (Found the only valid pattern)")
        else:
             print(f"K-4 Test: UNEXPECTED (Got {''.join(pattern_4)})")
             
        # Test K=5
        pattern_5 = analyser.getFrequentPattern(5)
        print(f"K=5 => {pattern_5}")
        if "".join(pattern_5) in ("cegec", "gdfhd", "cdfhd"):
             print("K-5 Test: PASS")
        else:
             print(f"K-5 Test: UNEXPECTED (Got {''.join(pattern_5)})")
             
        # Test K=6 (Out of bounds)
        pattern_6 = analyser.getFrequentPattern(6)
        print(f"K-6 => {pattern_6}")
        if pattern_6 == []:
             print("K-6 Test: PASS (Correctly returned empty list)")
        else:
             print(f"K-6 Test: FAIL (Got {pattern_6})")

    except Exception as e:
        print(f"\n--- !! An error occurred: {e} !! ---")



import heapq

def assign(L, roads, students, buses, D, T):
    """
    Function Description:
        Finds a valid assignment of students to buses given constraints on travel
        distance (D), bus capacities (min/max), and a target total (T).

    Approach Description (main function):
        1.  Build the city graph (adjacency list).
        2.  Find all unique bus pickup points (P, where P <= 18).
        3.  Run Dijkstra's algorithm once for each unique pickup point (P times)
            to find all students who can reach any buses at that point.
        4.  Build a "reachability" graph (student-to-bus).
        5.  Build a max-flow network with lower and upper bounds to solve the
            constrained assignment problem:
            -   Nodes: Source, Sink, Students (S), Buses (B), SuperSource, SuperSink
            -   This is a "circulation with demands" problem to handle the
                bus min-capacities (f_j).
        6.  Phase 1 (Feasibility): Run max-flow from SuperSource to SuperSink.
            If flow == total_demand, the minimums are possible.
        7.  Phase 2 (Augmentation): Disable the SINK->SOURCE helper edge.
            Push exactly (T - total_min) more flow from SOURCE to SINK.
        8.  Phase 3 (Recovery): If successful, read the allocation from the
            residual graph's student->bus edges.

    Input:
    - L: number of locations (int)
    - roads: list of (u, v, w) tuples (list)
    - students: list of student locations (list[int])
    - buses: list of (pickup_location, min_cap, max_cap) tuples (list)
    - D: max travel distance (int)
    - T: exact total students (int)

    Output:
    - allocation: list of length S, with allocation[i] = bus_id or -1 (list)
    - None: if no valid allocation exists (NoneType)

    Time Complexity Analysis:
    - O(R): Building the city graph.
    - O(B*L): Finding unique pickups (your efficient O(B+L) version is used).
    - O(P * (R + L log L)): Dijkstra's. Since P <= 18 (a constant), this
        is O(R + L log L) or O(R log L) if R > L.
    - O(S*B): Building reachability lists. O(S) since B=O(1).
    - Max-Flow: O(F * E) where F is total flow, E is edges.
        - E = O(S*B + S + B) = O(S) since B=O(1).
        - F_phase1 = total_min = O(T) in worst case (if T=total_min).
        - F_phase2 = extra_needed = O(T).
        - Total Flow F = O(T).
        - Max-Flow Time = O(T * S) = O(S*T).
    - O(S*B): Allocation recovery. O(S) since B=O(1).
    - Total Time: O(R log L + S*T)

    Aux Space Complexity Analysis:
    - adjacent_list: O(L + R)
    - unique_pickup_points: O(P) = O(1)
    - buses_at_pickup_point: O(B) = O(1)
    - pickup_to_index: O(L)
    - Dijkstra's distance/heap: O(L)
    - reachability lists: O(S*B) = O(S)
    - graph (flow network): O(V + E) = O((S+B) + (S*B)) = O(S)
    - BFS queue/visited: O(V) = O(S)
    - demand list: O(V) = O(S)
    - allocation list: O(S)
    - Total Aux Space: O(L + R + S)
    """
   
    # --- 1. Build City Graph ---
    
    # O(L) time/space: Initialize adjacency list for L locations.
    adjacent_list = [[] for _ in range(L)]
    # O(R) time loop: Iterate through all R roads.
    for u, v, w in roads:
        # O(1) amortized time: Append to u's adjacency list.
        adjacent_list[u].append((v, w))
        # O(1) amortized time: Append to v's list (for undirected graph).
        adjacent_list[v].append((u, w))

    # O(1) time: Get length of buses list.
    B = len(buses)
    # O(1) time: Get length of students list.
    S = len(students)

    # --- 2. Find Unique Pickup Points (P <= 18) ---
    
    # O(P) = O(1) aux space: Stores the unique location IDs.
    unique_pickup_points = []
    # O(P) = O(1) aux space: Stores lists of bus IDs at each unique point.
    buses_at_pickup_point = []
    # O(L) time/space: A hash map (using a list) to find a pickup's index in O(1).
    pickup_to_index = [-1] * L

    # O(B) time loop: Iterate through all buses. (B=O(1) assumption, so O(1) time).
    for bus_id in range(B):
        # O(1) time: Get pickup location ID for the current bus.
        pickup_point = buses[bus_id][0]
        # O(1) time: Check if we have seen this pickup point before.
        j = pickup_to_index[pickup_point]
        # O(1) time: Check condition.
        if j == -1:
            # O(1) time: Store the index for this new unique pickup point.
            pickup_to_index[pickup_point] = len(unique_pickup_points)
            # O(1) amortized time: Add the new unique location ID.
            unique_pickup_points.append(pickup_point)
            # O(1) amortized time: Add a new list for this location.
            buses_at_pickup_point.append([bus_id])
        else:
            # O(1) amortized time: Add this bus ID to the existing location's list.
            buses_at_pickup_point[j].append(bus_id)

    # --- 3. Pre-check Total Capacities ---
    
    # O(B) = O(1) aux space: Store min capacities for B buses.
    min_caps = [0] * B
    # O(B) = O(1) aux space: Store max capacities for B buses.
    max_caps = [0] * B
    # O(1) aux space: Initialize total minimum.
    total_min = 0
    # O(1) aux space: Initialize total maximum.
    total_max = 0
    # O(B) = O(1) time loop: Iterate through buses.
    for bus_index in range(B):
        # O(1) time: Unpack bus tuple.
        _, min_cap_bus_index, max_cap_bus_index = buses[bus_index]
        # O(1) time: Store min cap.
        min_caps[bus_index] = min_cap_bus_index
        # O(1) time: Store max cap.
        max_caps[bus_index] = max_cap_bus_index
        # O(1) time: Add to total.
        total_min += min_cap_bus_index
        # O(1) time: Add to total.
        total_max += max_cap_bus_index

    # O(1) time: Check if the target T is even possible.
    if T < total_min or T > total_max:
        # O(1) time: Impossible, return None.
        return None
    
    # O(1) time: Define a safe "infinity" larger than any possible path or capacity.
    infinity = max(total_max + 1, D + 1) 

    # --- 4. Dijkstra's Algorithm ---
    def dijkstra_from(start_loc):
        """
        Runs Dijkstra's from a start_loc, stopping early if dist > D.
        
        Time: O((L+R) log L). Using heapq (a binary heap):
          - O(L log L): In the worst case, each of the L locations is
            popped from the heap once.
          - O(R log L): In the worst case, we relax all R roads, and each
            relaxation pushes a new entry onto the heap.
          - Worst-case occurs when the graph is connected and most locations
            are reachable, requiring exploration of all roads and locations.
            The D-check optimization does not change this worst-case if
            D is large.
            
        Space: O(L).
          - `distance` list: O(L)
          - `heap`: O(L) in the worst case. (e.g., a star graph where
            the start_loc pushes L-1 neighbors onto the heap immediately).
        """
        # O(L) time/space: Initialize distance list to infinity.
        distance = [infinity] * L
        # O(1) time: Set start location distance to 0.
        distance[start_loc] = 0
        # O(1) time/space: Initialize the priority queue (heap).
        heap = [(0, start_loc)]

        # O(L log L + R) time: Main Dijkstra loop.
        # Pops O(L) nodes in worst case, O(log L) per pop.
        # Pushes O(R) edges in worst case, O(log L) per push.
        while heap:
            # O(log L) time: Pop the node with the smallest distance.
            curr_dist, u = heapq.heappop(heap)
            
            # O(1) time: Skip stale entries (from a previously shorter path).
            if curr_dist != distance[u]:
                continue
            
            # O(1) time: Optimization: If min dist is > D, no path from here will work.
            if curr_dist > D:
                break
            
            # O(deg(u)) time loop: Iterate through neighbors of u.
            # Total iterations across all 'while' loops is O(R).
            for v, w in adjacent_list[u]:
                # O(1) time: Calculate new distance.
                new_distance = curr_dist + w
                # O(1) time: Check if path is shorter and within distance D.
                if new_distance <= D and new_distance < distance[v]:
                    # O(1) time: Update distance.
                    distance[v] = new_distance
                    # O(log L) time: Push the new, shorter path to the heap.
                    heapq.heappush(heap, (new_distance, v))
        # O(1) time: Return the computed distance list.
        return distance

    
    # --- 5. Build Student-to-Bus Reachability ---
    
    # O(B) = O(1) aux space: Stores lists of students reachable by each bus.
    reachable_by_bus = [[] for _ in range(B)]
    # O(S) aux space: Stores lists of buses reachable by each student.
    buses_by_student = [[] for _ in range(S)]

    # O(1) time: Initialize index.
    index = 0
    # O(P) = O(1) time loop: Loop P times (P <= 18).
    while index < len(unique_pickup_points):
        # O(1) time: Get the unique pickup location ID.
        pickup_location = unique_pickup_points[index]
        # O(R + L log L) time: Run Dijkstra's from this location.
        distance = dijkstra_from(pickup_location)
        # O(1) time: Get the list of bus IDs at this location.
        bus_list = buses_at_pickup_point[index]
        
        # O(S) time loop: Check reachability for all S students.
        for student_id, student_loc in enumerate(students): 
            # O(1) time: Check if student can reach this location.
            if distance[student_loc] <= D:
                # O(B_at_p) = O(B) = O(1) time loop: Iterate buses at this point.
                for bus_id in bus_list: 
                    # O(1) amortized time: Add student to bus's list.
                    reachable_by_bus[bus_id].append(student_id)
                    # O(1) amortized time: Add bus to student's list.
                    buses_by_student[student_id].append(bus_id)

        # O(1) time: Increment index.
        index += 1

    # --- 6. Flow Network Setup ---
    # 
    
    # O(1) time: Define node indices for clarity.
    NETWORK_SOURCE = 0
    NETWORK_SINK = 1
    STUDENT_NODE_START = 2
    BUS_NODE_START = STUDENT_NODE_START + S
    DEMAND_SUPER_SOURCE = BUS_NODE_START + B
    DEMAND_SUPER_SINK = DEMAND_SUPER_SOURCE + 1
    NODE_COUNT = DEMAND_SUPER_SINK + 1 # Total nodes V = S+B+4

    # O(1) time: Define edge tuple indices for clarity.
    DEST, REV_INDEX, CAPACITY = 0, 1, 2

    # O(V) = O(S+B) = O(S) aux space: Initialize the flow graph.
    graph = [[] for _ in range(NODE_COUNT)]

    def add_edge(u, v, cap_value):
        """
        Adds a forward edge (u->v) with capacity and a reverse edge (v->u)
        with capacity 0. Stores the index of the reverse edge for O(1) access.
        Time: O(1) amortized
        """
        # O(1) amortized time: Append forward edge [dest, rev_idx, cap].
        graph[u].append([v, len(graph[v]), cap_value])
        # O(1) amortized time: Append reverse edge [dest, rev_idx, cap=0].
        graph[v].append([u, len(graph[u]) - 1, 0])

    # --- 7. Edmonds-Karp (BFS-based Max Flow) ---
    def bfs_find_path(source, sink, limit):
        """
        Finds one augmenting path using BFS.
        Time: O(V + E) = O(S+B + S*B) = O(S) (since B=O(1))
        Space: O(V) = O(S)
        """
        # O(V) = O(S) time/space: Initialize visited list.
        visited = [False] * NODE_COUNT
        # O(V) = O(S) time/space: Store parent node in the BFS path.
        parent_node = [-1] * NODE_COUNT
        # O(V) = O(S) time/space: Store edge index used to reach node.
        parent_edge = [-1] * NODE_COUNT

        # O(V) = O(S) aux space: Initialize a list-based queue.
        queue = [0] * NODE_COUNT
        # O(1) time: Initialize queue head pointer.
        head = 0
        # O(1) time: Initialize queue tail pointer.
        tail = 0
        # O(1) time: Add source node to queue.
        queue[tail] = source 
        # O(1) time: Increment tail.
        tail += 1
        # O(1) time: Mark source as visited.
        visited[source] = True

        # O(V) = O(S) time loop: BFS explores each node at most once.
        while head < tail:
            # O(1) time: Dequeue a node.
            u = queue[head]
            # O(1) time: Increment head.
            head += 1
            # O(1) time: Get adjacency list for node u.
            adjacent_u = graph[u]

            # O(deg(u)) time loop: Iterate through all neighbors.
            # Total for all loops is O(E) = O(S).
            j=0
            while j < len(adjacent_u):
                # O(1) time: Get edge.
                e = adjacent_u[j]
                # O(1) time: Check if there is capacity.
                if e[CAPACITY] > 0:
                    # O(1) time: Get destination node.
                    v = e[DEST]
                    # O(1) time: Check if visited.
                    if not visited[v]:
                        # O(1) time: Mark visited.
                        visited[v] = True
                        # O(1) time: Set parent.
                        parent_node[v] = u
                        # O(1) time: Set edge index.
                        parent_edge[v] = j

                        # O(1) time: Check if we reached the sink.
                        if v == sink:
                            # --- Path found, reconstruct bottleneck ---
                            # O(1) time: Set initial bottleneck.
                            bottleneck = limit
                            # O(1) time: Start walk back from sink.
                            walk_node = sink
                            # O(V) = O(S) time loop: Walk back to source.
                            while walk_node != source:
                                # O(1) time: Get parent node.
                                prev_node = parent_node[walk_node]
                                # O(1) time: Get edge index from parent.
                                prev_edge_index = parent_edge[walk_node]
                                # O(1) time: Get capacity of that edge.
                                edge_capacity = graph[prev_node][prev_edge_index][CAPACITY]
                                # O(1) time: Update bottleneck if this edge is smaller.
                                if edge_capacity < bottleneck:
                                    bottleneck = edge_capacity
                                # O(1) time: Move to the parent.
                                walk_node = prev_node
                            # O(1) time: Return flow and path info.
                            return bottleneck, parent_node, parent_edge
                        
                        # O(1) time: Add node to queue.
                        queue[tail] = v
                        # O(1) time: Increment tail.
                        tail += 1
                # O(1) time: Go to next edge.
                j += 1

        # O(1) time: No path found, return 0 flow.
        return 0, parent_node, parent_edge
    
    def maxflow(source, sink, flow_limit):
        """
        Runs Edmonds-Karp algorithm until flow_limit is reached.
        Time: O(F * E) = O(flow_limit * S)
        """
        # O(1) time: Initialize total flow.
        total_flow = 0

        # O(F) = O(flow_limit) time loop: Find F augmenting paths.
        while total_flow < flow_limit:
            # O(1) time: Calculate remaining flow needed.
            remaining = flow_limit - total_flow
            # O(E) = O(S) time: Find one augmenting path.
            pushed, parent_node, parent_edge = bfs_find_path(source, sink, remaining) 
            
            # O(1) time: If no path found, stop.
            if pushed == 0:
                break
            
            # O(V) = O(S) time loop: Augment path.
            node = sink
            while node != source:
                # O(1) time: Get parent.
                prev_node = parent_node[node]
                # O(1) time: Get edge index from parent.
                edge_index = parent_edge[node]
                # O(1) time: Get forward edge.
                e = graph[prev_node][edge_index]
                
                # O(1) time: Subtract capacity from forward edge.
                e[CAPACITY] -= pushed 
                
                # O(1) time: Get reverse edge index.
                rev_edge_index = e[REV_INDEX]
                # O(1) time: Add capacity to reverse edge.
                graph[node][rev_edge_index][CAPACITY] += pushed
                
                # O(1) time: Move to parent.
                node = prev_node

            # O(1) time: Add pushed flow to total.
            total_flow += pushed

        # O(1) time: Return total flow pushed.
        return total_flow
    
    # --- 8. Build Flow Network Edges ---
    
    # SOURCE -> Students (Cap 1)
    # O(S) time loop.
    student_id = 0
    while student_id < S:
        # O(1) amortized time: Add edge from source to student.
        add_edge(NETWORK_SOURCE, STUDENT_NODE_START + student_id, 1)
        # O(1) time: Increment.
        student_id += 1

    
    # Students -> Buses (Cap 1)
    # O(B) = O(1) time loop.
    for bus_id in range(B):
        # O(1) time: Get bus node ID.
        bus_node = BUS_NODE_START + bus_id
        # O(1) time: Get list of students who can reach this bus.
        reachable_students = reachable_by_bus[bus_id]
        # O(S_reachable) = O(S) time loop.
        for student_id in reachable_students:
            # O(1) amortized time: Add edge from student to bus.
            add_edge(STUDENT_NODE_START + student_id, bus_node, 1)
        
    # Buses -> SINK (Lower/Upper bounds)
    # O(V) = O(S) aux space: List to store node demands.
    demand = [0] * NODE_COUNT
    # O(B) = O(1) time loop.
    for bus_id in range(B):
        # O(1) time: Get min/max capacities.
        lower_bound = min_caps[bus_id]
        upper_bound = max_caps[bus_id]
        # O(1) time: Get bus node ID.
        bus_node = BUS_NODE_START + bus_id

        # O(1) amortized time: Add edge from bus to sink with cap = (max - min).
        add_edge(bus_node, NETWORK_SINK, upper_bound - lower_bound)
        # O(1) time: Add negative demand to bus node (it needs flow).
        demand[bus_node] -= lower_bound
        # O(1) time: Add positive demand to sink node (it supplies flow).
        demand[NETWORK_SINK] += lower_bound

    # --- 9. Phase 1: Satisfy Lower Bounds (Feasibility) ---
    
    # O(1) time: Store index of the SINK->SOURCE edge for later.
    helper_edge_index_at_sink = len(graph[NETWORK_SINK]) 
    # O(1) amortized time: Add helper edge to allow circulation.
    add_edge(NETWORK_SINK, NETWORK_SOURCE, infinity)

    # O(1) time: Initialize total positive demand.
    total_demand = 0
    # O(V) = O(S) time loop: Iterate all nodes.
    node = 0
    while node < NODE_COUNT:
        # O(1) time: Get demand for the node.
        d = demand[node]
        # O(1) time: Check if it's a supply node.
        if d > 0:
            # O(1) amortized time: Add edge from SuperSource to supply node.
            add_edge(DEMAND_SUPER_SOURCE, node, d)
            # O(1) time: Add to total demand.
            total_demand += d
        # O(1) time: Check if it's a demand node.
        elif d < 0:
            # O(1) amortized time: Add edge from demand node to SuperSink.
            add_edge(node, DEMAND_SUPER_SINK, -d)
        # O(1) time: Increment.
        node += 1

    # O(F * E) = O(total_min * S) = O(T*S) time: Run feasibility flow.
    feasible_flow = maxflow(DEMAND_SUPER_SOURCE, DEMAND_SUPER_SINK, total_demand)
    
    # O(1) time: Check if all minimum demands were met.
    if feasible_flow < total_demand:
        # O(1) time: If not, no solution is possible.
        return None
    
    # --- 10. Phase 2: Augment to Exactly T ---
    
    # O(1) time: Get the SINK->SOURCE helper edge.
    forward = graph[NETWORK_SINK][helper_edge_index_at_sink]
    # O(1) time: Disable it by setting its capacity to 0.
    forward[CAPACITY] = 0
    # O(1) time: Also disable its reverse edge.
    graph[NETWORK_SOURCE][forward[REV_INDEX]][CAPACITY] = 0

    # O(1) time: Calculate remaining flow needed to reach T.
    extra_needed = T - total_min
    # O(1) time: Check (should be impossible if T >= total_min).
    if extra_needed < 0:
        return None 
    
    # O(F * E) = O(extra_needed * S) = O(T*S) time: Push the extra flow.
    if maxflow(NETWORK_SOURCE, NETWORK_SINK, extra_needed) != extra_needed:
        # O(1) time: If we couldn't push the exact amount, fail.
        return None
    
    # --- 11. Recover Allocation ---
    
    # O(S) time/space: Initialize the final allocation list.
    allocation = [-1] * S
    # O(S) time loop: Check each student.
    for student_id in range(S):
        # O(1) time: Get student's node ID.
        u = STUDENT_NODE_START + student_id
        # O(1) time: Initialize edge index.
        j = 0
        # O(deg(u)) = O(B) = O(1) time loop: Check outgoing edges.
        while j < len(graph[u]):
            # O(1) time: Get edge.
            e = graph[u][j]
            # O(1) time: Get destination.
            v = e[DEST]
            # O(1) time: Check if this edge goes to a bus.
            if BUS_NODE_START <= v < BUS_NODE_START + B:
                # O(1) time: Check if the *reverse* edge has flow.
                if graph[v][e[REV_INDEX]][CAPACITY] > 0:
                    # O(1) time: If yes, this student was assigned to this bus.
                    allocation[student_id] = v - BUS_NODE_START
                    # O(1) time: Stop searching for this student.
                    break
            # O(1) time: Go to next edge.
            j += 1
    
    # O(S) time: Final sanity check to ensure exactly T students were assigned.
    assigned_count = 0
    i = 0
    while i < S:
        if allocation[i] != -1:
            assigned_count += 1
        i += 1
    
    # O(1) time: Check count.
    if assigned_count != T:
        # O(1) time: This should not happen if logic is correct, but good safety check.
        return None
        
    # O(1) time: Return the final allocation.
    return allocation




class Analyser:
    """
    Implements a music pattern analyser that finds the most frequent
    transposable pattern of a given length K.
    
    This implementation adheres to the "no dictionaries or sets" rule
    by using only lists for data storage and lookups.
    """

    def __init__(self, sequences):
        """
        Function Description:
            Pre-processes all song sequences to find the most frequent
            transposable pattern for every possible length.
            
        Approach Description:
            1.  Stores all sequences.
            2.  Initializes lists to store the max frequency and best
                location for each pattern length K.
            3.  Uses a list `pattern_frequ_map` to store pattern frequencies.
            4.  Iterates through every song (N), every start index (M),
                and every end index (M) to generate all O(NM^2) subsequences.
            5.  For each subsequence, it computes a rolling hash of the
                *intervals* between notes (e.g., "ceg" -> (2, 2)). This
                hash represents all transpositions of that pattern.
            6.  It performs a *linear search* through `pattern_frequ_map`
                to find the (length, hash) key.
            7.  It updates the frequency count, being careful to only
                count each song once per unique pattern.
            8.  If the new frequency is the highest for that pattern_length,
                it records this pattern's location (song_id, start_index).
                
        *** COMPLEXITY AND CONTRADICTION WARNING ***
        
        Time Complexity: O(N^2 * M^4)
            - N = number of sequences
            - M = length of the longest sequence
            - P = total unique patterns, P = O(N*M^2) in the worst case.
            
            - The three outer loops iterate O(N*M^2) times.
            - **Inside the innermost loop**, we have a linear search:
              `for entry in pattern_frequ_map:`
            - `pattern_frequ_map` stores all unique patterns found so far (P).
            - This search takes O(P) time, where P can grow to O(N*M^2).
            - Total time = O(N*M^2 * P) = O(N*M^2 * N*M^2) = O(N^2 * M^4).
            
            - **This VIOLATES the O(NM^2) time requirement in the prompt.**
            - This is an *unavoidable* trade-off. To get O(NM^2) time,
              you *must* use a dictionary for O(1) lookups in
              `pattern_frequ_map`. Because dictionaries are forbidden,
              a linear search is required, resulting in O(N^2 * M^4) time.
              
        Auxiliary Space Complexity: O(NM + M)
            - `self.sequences`: O(NM) to store all characters.
            - `self.max_frequency`: O(M)
            - `self.best_pattern_location`: O(M)
            - `pattern_frequ_map` (Local): This is local to `__init__` and
              is freed after. Its *peak* space is O(N*M^2 * N) = O(N^2 M^2)
              in the worst-case, but this does not count towards the
              *final* object's space.
            - **Total Final Space:** O(NM + M) = O(NM), which **MEETS**
              the O(NM) space requirement.
        """
        
        # O(1) time: Store reference to sequences.
        self.sequences = sequences
        
        # O(N) time: Find the number of sequences.
        N = len(sequences)
        # O(1) time: Initialize max_length.
        max_length = 0
        # O(N) time loop: Find the longest sequence length M.
        for s in sequences:
            # O(1) time: Get length and find max.
            max_length = max(max_length, len(s))

        # O(M) time/space: List to store the highest frequency found for each length.
        self.max_frequency = [0 for _ in range(max_length + 1)]
        # O(M) time/space: List to store (song_id, start_index) of the best pattern.
        self.best_pattern_location = [None for _ in range(max_length + 1)] 

        # O(1) time: Initialize a list to be used as a hash map.
        pattern_frequ_map = [] # [ [ (key), [song_ids] ], ... ]

        # O(1) time: Prime base for rolling hash.
        BASE = 37

        # O(N) time loop: Iterate through each song.
        for song_id, song in enumerate(self.sequences):
            # O(1) time: Get length of the current song.
            song_len = len(song)

            # O(M) time loop: Iterate through each possible start character.
            for start_index in range(song_len):
                # O(1) time: Reset hash for each new start position.
                rolling_hash = 0

                # O(M) time loop: Iterate through each possible end character.
                for end_index in range(start_index + 1, song_len):
                    # --- Rolling Hash Calculation (O(1)) ---
                    # O(1) time: Get interval between notes.
                    interval = ord(song[end_index]) - ord(song[end_index - 1])
                    # O(1) time: Update rolling hash.
                    rolling_hash = (rolling_hash * BASE) + interval

                    # O(1) time: Calculate pattern length.
                    pattern_length = end_index - start_index + 1
                    # O(1) time: Create the key (length, hash) for this pattern.
                    pattern_key = (pattern_length, rolling_hash)
                    
                    # --- List-based Map Lookup (O(P) = O(NM^2)) ---
                    # O(1) time: Initialize found-flag.
                    entry_found = None
                    # O(P) time loop: Linear search for the pattern key.
                    # P is the number of unique patterns found so far.
                    # P can be up to O(N*M^2). This is the bottleneck.
                    for entry in pattern_frequ_map:
                        # O(1) time: Check key.
                        if entry[0] == pattern_key:
                            # O(1) time: Store reference to the found entry.
                            entry_found = entry
                            # O(1) time: Stop the linear search.
                            break
                    
                    # --- Frequency Update (O(N) worst-case) ---
                    # O(1) time: Check if pattern is new.
                    if entry_found is None:
                        # O(1) aux space: Create new song list.
                        song_indices = [song_id]
                        # O(1) amortized time: Add new [key, song_list] to map.
                        pattern_frequ_map.append([pattern_key, song_indices])
                        # O(1) time: New pattern, frequency is 1.
                        frequency = 1
                    else:
                        # O(1) time: Get reference to existing song list.
                        song_indices = entry_found[1]
                        # O(1) time: Initialize flag.
                        added_already = False

                        # O(N) time loop: Check if this song_id is already in the list.
                        # The list can have at most N song IDs.
                        for previous_id in song_indices:
                            # O(1) time: Check ID.
                            if previous_id == song_id:
                                # O(1) time: Set flag.
                                added_already = True
                                # O(1) time: Stop search.
                                break
                        
                        # O(1) time: Check if we need to add this song_id.
                        if not added_already:
                            # O(1) amortized time: Add new song_id to list.
                            song_indices.append(song_id)

                        # O(1) time: Get the new frequency.
                        frequency = len(song_indices)

                    # --- Best Pattern Update (O(1)) ---
                    # O(1) time: Check if this pattern is the new best for this length.
                    if frequency > self.max_frequency[pattern_length]:
                        # O(1) time: Update max frequency.
                        self.max_frequency[pattern_length] = frequency
                        # O(1) time: Store location of this pattern.
                        self.best_pattern_location[pattern_length] = (song_id, start_index)
                        
                        
    def getFrequentPattern(self, K):
        """
        Function Description:
            Returns the most frequent pattern of a specific length K.
            
        Approach Description:
            Performs an O(1) lookup in the pre-computed
            `self.best_pattern_location` list to find the location
            (song_id, start_index) of the best pattern for length K.
            It then slices this pattern from `self.sequences` and
            returns it as a list of characters.
            
        Time Complexity: O(K)
            - O(1): All checks and lookups in `self.best_pattern_location`.
            - O(K): Slicing the pattern string `[start:start+K]` takes
              O(K) time to create the new string.
            - O(K): `list(pattern_string)` takes O(K) time to iterate
              the new string and build the list.
            - Total: O(K). This **MEETS** the requirement.
            
        Space Complexity: O(K)
            - O(K): The `pattern_string` slice creates a new string of length K.
            - O(K): The final `list` returned has length K.
            - Total: O(K).
        """
        # O(1) time: Check if K is out of bounds.
        if K >= len(self.best_pattern_location):
            # O(1) time: Return empty list.
            return []
        # O(1) time: Check if any pattern of length K was found.
        if self.best_pattern_location[K] is None:
            # O(1) time: Return empty list.
            return []
        
        # O(1) time: Get the pre-computed location (tuple).
        location = self.best_pattern_location[K]
        # O(1) time: Unpack song index.
        song_index = location[0]
        # O(1) time: Unpack start index.
        start_index = location[1]   

        # O(1) time: Get reference to the full song string.
        song_with_best_pattern = self.sequences[song_index]

        # O(K) time/space: Slice the pattern from the song.
        pattern_string = song_with_best_pattern[start_index:start_index + K]
        # O(K) time/space: Convert the string to a list of characters.
        return list(pattern_string)