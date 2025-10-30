import heapq

def assign(L, roads, students, buses, D, T):
    """
    Function Purpose:
Determines a valid allocation of students to buses under the following constraints:
-Each student can only walk up to distance D to reach a bus.
-Each bus has a minimum and maximum capacity.
-Exactly T students must be assigned in total.

Algorithm Overview:
1. Construct the City Graph- Represent the city as an adjacency list, where locations are nodes and roads are weighted edges.

2.Identify Unique Pickup Points:
-Extract all unique pickup locations from the list of buses (P, where P<=18)
-Group buses together by their pickup location for efficient distance queries

3.Using Dijkstra's Algorithm Reacability is calculated:
- For each unique pickup point, run Dijkstra's algorithm to determine which students can reach that point within distance D.
    Build two maps:
    -For each bus (B), the students (S) who can reach it wihtin distance D
    -For each student (S), the buses (B) they can reach woithin distance D

4.Build a max-flow network with lower and upper bounds to solve problem:
            -   Nodes: Source, Sink, Students (S), Buses (B), SuperSource, SuperSink
            -   This is a “circulation with demands” problem, which ensures that every bus's minimum required capacity (f_j) is satisfied, 
                where f_j represents the minimum number of students that bus j must carry.

5. Feasibility Flow (Circulation with Demands):
-Ensure that all bus minimum capacities can be satisfied
- run max-flow from Super source to Super sink.
- If the flow is less than total demand, a valid allocation is impossible.
- if flow == total demand,minimum possible, proceed to Augmentation Flow.

6.Augmentayion Flow (Exact Assignment Count):
- Disable the helper edge (sink → source) used during circulation check.
- Attempt to push exactly T - total_min additional flow from the original source to sink.
- Return None, if fails

7. Extract Assignments:
-Traverse the residual graph to determine which student was assigned to which bus
-Return a list where allocation[i] is the bus ID assigned to student i, or -1 if unassigned.
    
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
    - O(R): Building the `adjacent_list`, where R is the number of roads.
    - O(L + B): Finding unique pickups, by initialising a list of size L (O(L))
        and iterating through B buses (O(B)).
    - O(P * (R + L log L)): Running Dijkstra's from P unique pickup points.
        Each Dijkstra's is O(R + L log L). Since P <= 18 (a constant),
        this simplifies to O(R + L log L).
    - O(S*B): Building reachability lists. Since B is a constant (B=O(1)),
        this simplifies to O(S).
    - Max-Flow (Edmonds-Karp): O(F * E) where F is total flow, E is edges.
        - E (Edges) = O(S*B + S + B). Since B=O(1), E = O(S).
        - F (Flow) = Feasibility Flow + Augmentation Flow.
        - Feasibility Flow  = total_min. In the worst case, total_min can be O(T).
        - Augmentation Flow = extra_needed = T - total_min. This is also O(T).
        - Total Flow F = O(T).
        - Max-Flow Time = O(T * S) = O(S*T).
    - O(S*B): Allocation recovery. Since B=O(1), this simplifies to O(S).
    - Total Time: The sum of these parts is O(R) + O(L+B) + O(R + L log L) + O(S) + O(S*T) + O(S).
    - Since L+R log L dominates L and R, and B=O(1), and S*T is separate,
        the final worst-case time is O(S*T + L + R log L).


    Aux Space Complexity Analysis:
    The worst-case auxiliary space is O(S + L + R), based on the following:
    - City Graph: `adjacent_list` requires O(L) space for the list of lists and O(R)
        space to store all the road tuples, for a total of O(L + R).
    - Dijkstra's & Helpers:
        - `pickup_to_index`: Requires a list of size L, which is O(L).
        - `dijkstra_from` internals: The `distance` list is O(L) and the `heap`
          is O(L) in the worst case.
        - `unique_pickup_points` and `buses_at_pickup_point`: These are O(P) and O(B)
          respectively. Since P <= 18 and B is assumed to be O(1), these are O(1).
    - Reachability Lists:
        - `buses_by_student`: In the worst case, every student can reach every bus,
          requiring O(S * B) space. Based on the B=O(1) assumption, this is O(S).
        - `reachable_by_bus`: This is O(S * B), which simplifies to O(S).
    - Flow Network ( largest component):
        - `graph` (adjacency list): Stores O(V) nodes and O(E) edges.
          - V = O(S + B) = O(S) (since B=O(1)).
          - E = O(S + S*B + B) = O(S) (since B=O(1)).
          - Total space for `graph` is O(V + E) = O(S).
        - `bfs_find_path` internals: `visited`, `parent_node`, `parent_edge`, and `queue`
          are all lists of size `NODE_COUNT` (which is O(V)), so they are O(S).
        - `demand` list: This is size `NODE_COUNT`, so it is O(S).
    - Final Output:
        - `allocation`: This is a list of size S, which is O(S).
    - Total: The total space is the sum of these parts: O(L+R) + O(L) + O(S) + O(S) + O(S).
    - The dominant terms are O(L + R + S).
    """

    # 1. Build city graph # 

    adjacent_list = [[] for _ in range(L)] # O(L) time/space - intilaises adjacency list for L locations
    for u, v, w in roads:  # O(R) time loop - iterate through R roads
        adjacent_list[u].append((v, w)) #O(1) time- appends u's adjacent list
        adjacent_list[v].append((u, w)) #O(1) time- appends v's adjacent list

    B = len(buses)   # O(1) time - number of buses list
    S = len(students) #O(1) time - number of students list



    # 2. finds unique pickup points (P<=18) #
    unique_pickup_points = []  #O(P) = O(1) aux space as it stores unique location IDS
    buses_at_pickup_point = [] # O(P) = O(1) aux spacw as it stores list of bus IDS at each of the unique point
    pickup_to_index = [-1] * L # O(L) time and space; a has map using a list is used to find pickup index in O(1)


    for bus_id in range(B):
        pickup_point = buses[bus_id][0]
        j = pickup_to_index[pickup_point]
        if j == -1:
            pickup_to_index[pickup_point] = len(unique_pickup_points)
            unique_pickup_points.append(pickup_point)
            buses_at_pickup_point.append([bus_id])
        else:
            buses_at_pickup_point[j].append(bus_id)


    min_caps = [0] * B
    max_caps = [0] * B
    total_min = 0
    total_max = 0
    for bus_index in range(B):
        _, min_cap_bus_index, max_cap_bus_index = buses[bus_index]
        min_caps[bus_index] = min_cap_bus_index
        max_caps[bus_index] = max_cap_bus_index
        total_min += min_cap_bus_index
        total_max += max_cap_bus_index

    if T < total_min or T > total_max:
        return None
    

    infinity = max(total_max + 1, D + 1) 

    def dijkstra_from(start_loc):
        distance = [infinity] * L
        distance[start_loc] = 0
        heap = [(0, start_loc)]

        while heap:
            curr_dist, u = heapq.heappop(heap)
            if curr_dist != distance[u]:
                continue
            if curr_dist > D:
                break
            for v, w in adjacent_list[u]:
                new_distance = curr_dist + w
                if new_distance <= D and new_distance < distance[v]:
                    distance[v] = new_distance
                    heapq.heappush(heap, (new_distance, v))
        return distance

    
    #reachability
    reachable_by_bus = [[] for _ in range(B)]
    buses_by_student = [[] for _ in range(S)]

    index = 0
    while index < len(unique_pickup_points):
        pickup_location = unique_pickup_points[index]
        distance = dijkstra_from(pickup_location)
        bus_list = buses_at_pickup_point[index]
        
       
        for student_id, student_loc in enumerate(students): 
            if distance[student_loc] <= D:
                for bus_id in bus_list:
                    reachable_by_bus[bus_id].append(student_id)
                    buses_by_student[student_id].append(bus_id)

        index += 1

    # flow network
    NETWORK_SOURCE = 0
    NETWORK_SINK = 1
    STUDENT_NODE_START = 2
    BUS_NODE_START = STUDENT_NODE_START + S
    DEMAND_SUPER_SOURCE = BUS_NODE_START + B
    DEMAND_SUPER_SINK = DEMAND_SUPER_SOURCE + 1
    NODE_COUNT = DEMAND_SUPER_SINK + 1

    DEST, REV_INDEX, CAPACITY = 0, 1, 2

    graph = [[] for _ in range(NODE_COUNT)]

    def add_edge(u, v, cap_value):
        # forward edge
        graph[u].append([v, len(graph[v]), cap_value])
        # backward edge
        graph[v].append([u, len(graph[u]) - 1, 0])

    def bfs_find_path(source, sink, limit):
        visited = [False] * NODE_COUNT
        parent_node = [-1] * NODE_COUNT
        parent_edge = [-1] * NODE_COUNT

        queue = [0] * NODE_COUNT
        head = 0
        tail = 0
        queue[tail] = source 
        tail += 1
        visited[source] = True

        while head < tail:
            u = queue[head]
            head += 1
            adjacent_u = graph[u]

            j=0
            while j < len(adjacent_u):
                e = adjacent_u[j]
                if e[CAPACITY] > 0:
                    v = e[DEST]
                    if not visited[v]:
                        visited[v] = True
                        parent_node[v] = u
                        parent_edge[v] = j

                        if v == sink:
                            bottleneck = limit
                            walk_node = sink
                            while walk_node != source:
                                prev_node = parent_node[walk_node]
                                prev_edge_index = parent_edge[walk_node]
                                edge_capacity = graph[prev_node][prev_edge_index][CAPACITY]
                                if edge_capacity < bottleneck:
                                    bottleneck = edge_capacity
                                walk_node = prev_node
                            return bottleneck, parent_node, parent_edge
                        
                        queue[tail] = v
                        tail += 1
                j += 1

        return 0, parent_node, parent_edge
    
    def maxflow(source, sink, flow_limit):
        total_flow = 0

        while total_flow < flow_limit:
            remaining = flow_limit - total_flow
            pushed, parent_node, parent_edge = bfs_find_path(source, sink, remaining)
            if pushed == 0:
                break
            
            node = sink
            while node != source:
                prev_node = parent_node[node]
                edge_index = parent_edge[node]
                e = graph[prev_node][edge_index]
                e[CAPACITY] -= pushed
                rev_edge_index = e[REV_INDEX]
                graph[node][rev_edge_index][CAPACITY] += pushed
                node = prev_node

            total_flow += pushed

        return total_flow
    
    student_id = 0
    while student_id < S:
        add_edge(NETWORK_SOURCE, STUDENT_NODE_START + student_id, 1)
        student_id += 1

    
    # student -> bus
    for bus_id in range(B):
        bus_node = BUS_NODE_START + bus_id
        reachable_students = reachable_by_bus[bus_id]
        for student_id in reachable_students:
            add_edge(STUDENT_NODE_START + student_id, bus_node, 1)
        
    # bus -> sink 
    demand = [0] * NODE_COUNT
    for bus_id in range(B):
        lower_bound = min_caps[bus_id]
        upper_bound = max_caps[bus_id]
        bus_node = BUS_NODE_START + bus_id

        add_edge(bus_node, NETWORK_SINK, upper_bound - lower_bound)
        demand[bus_node] -= lower_bound
        demand[NETWORK_SINK] += lower_bound

    helper_edge_index_at_sink = len(graph[NETWORK_SINK]) 
    add_edge(NETWORK_SINK, NETWORK_SOURCE, infinity)

    total_demand = 0
    node = 0
    while node < NODE_COUNT:
        d = demand[node]
        if d > 0:
            add_edge(DEMAND_SUPER_SOURCE, node, d)
            total_demand += d
        elif d < 0:
            add_edge(node, DEMAND_SUPER_SINK, -d)
        node += 1

    feasible_flow = maxflow(DEMAND_SUPER_SOURCE, DEMAND_SUPER_SINK, total_demand)
    if feasible_flow < total_demand:
        return None
    
    forward = graph[NETWORK_SINK][helper_edge_index_at_sink]
    forward[CAPACITY] = 0
    graph[NETWORK_SOURCE][forward[REV_INDEX]][CAPACITY] = 0


    extra_needed = T - total_min
    if extra_needed < 0:
        return None 
    if maxflow(NETWORK_SOURCE, NETWORK_SINK, extra_needed) != extra_needed:
        return None
        
    allocation = [-1] * S
    for student_id in range(S):
        u = STUDENT_NODE_START + student_id
        j = 0
        while j < len(graph[u]):
            e = graph[u][j]
            v = e[DEST]
            if BUS_NODE_START <= v < BUS_NODE_START + B:
                if graph[v][e[REV_INDEX]][CAPACITY] > 0:
                    allocation[student_id] = v - BUS_NODE_START
                    break    
            j += 1  
            
    assigned_count = 0
    i = 0
    while i < S:
        if allocation[i] != -1:
            assigned_count += 1
        i += 1
    if assigned_count != T:
        return None
        
    return allocation




##QUESTION 2##
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
    
    
