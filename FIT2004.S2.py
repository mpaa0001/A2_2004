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


    for bus_id in range(B):     # O(B) time loop; iterates through all buses. With the assumtpion (B=O(1)), therefore takes O(1) time
        pickup_point = buses[bus_id][0]  #O(1) time; gets the pickup location ID for the current bus
        j = pickup_to_index[pickup_point] #O(1) time; checks if the pickup oint has been seen before
        if j == -1:   #O(1) time as it is a check cnodition
            pickup_to_index[pickup_point] = len(unique_pickup_points)  # O(1) time; is stores index for this new pickup point
            unique_pickup_points.append(pickup_point) # O(1) time; adds new unniuqe location ID
            buses_at_pickup_point.append([bus_id])  #O(1) time; add new list for this location
        else:
            buses_at_pickup_point[j].append(bus_id)  #O(1) time; add this bus ID to exisitng locations list

    #checking total capacities
    min_caps = [0] * B # O(B) = O(1) aux space; stores min capacities for B buses
    max_caps = [0] * B #O(B) = O(1) aux space; sotres max capacities for B buses
    total_min = 0 #O(1) aux space; intalslies the total minimum
    total_max = 0 #O(1) aux spacw; intialsies total maximum
    for bus_index in range(B): # O(B) = O(1) time loop; iterates through the buses
        _, min_cap_bus_index, max_cap_bus_index = buses[bus_index] # O(1) time; bus tuple unpacked
        min_caps[bus_index] = min_cap_bus_index #O(1) time; sotres minimum capacity
        max_caps[bus_index] = max_cap_bus_index #O(1) time; sotres maximum capacity
        total_min += min_cap_bus_index #O(1) time; add to total
        total_max += max_cap_bus_index # O(1) time; add to total

    # O(1) time; chekcs if target T is possible
    if T < total_min or T > total_max:
        return None #O(1) timee; impossiple, will return none
    

    infinity = max(total_max + 1, D + 1)  #O(1) time; a safe infinity larger than any possible path or capacity



    # 3. Dijkstra's algorithm
    def dijkstra_from(start_loc):
        """
        Runs Dijktra's from the start_loc, stopping ealry if the curr_dist  > D
        Time: O((L+R) log L). Using heapq (binary heap)
        - O(L log L): in the worst case, where each of the L locations is popped from heap once
        - O(R log L): in worst case, relax all R roads and each relaxation pushes new entry onto heap
        - Worst case happens when graph is connected and most locations are reachable therefore requiring exploaration of 
        all the roads and locations

        Space: O(L)
        - 'distance' list : O(L)
        - 'heap' : O(L) in worst case
        
        """
        distance = [infinity] * L #O(L) time and space; intialises disantce list to infinty
        distance[start_loc] = 0 #O(1) time: set start locaiton to distance to 0
        heap = [(0, start_loc)] # O(1) time and space: intialises priority queue (heap)


        #O(L log L + R) time: main Dijkstra loop
        #Pops O(L) nodes in worst case, O(log L) per pop
        # Pushes O(R) edges in worst case, O (log L ) per push 
        while heap:
            curr_dist, u = heapq.heappop(heap) # O(log L) time: pop node with smallest distance
            if curr_dist != distance[u]: # O(1) time: stale entries skipped
                continue
            if curr_dist > D: # O(1 ) time: optimisation; if minimum distance is > D, no path from here will work
                break

            # Total iterations acorss all 'while' loops is O(R)
            for v, w in adjacent_list[u]: 
                new_distance = curr_dist + w # O(1) time: calcualte new distance
                if new_distance <= D and new_distance < distance[v]: # O(1) time; check if path is shorter and within distance D
                    distance[v] = new_distance # O(1) time: updates distance
                    heapq.heappush(heap, (new_distance, v)) # O(log L) time: push new and shorter path to heap
        return distance # O(1) time: returns computed distance list

    
    # Student to Bus Reachability 

    # O(B) = O(1) aux space: stores lists of studetns reachable for each bus
    reachable_by_bus = [[] for _ in range(B)]
    # O(S) aux space : stores lists of buses reachable by each student
    buses_by_student = [[] for _ in range(S)]

    # O(1) time: intialsies index
    index = 0
    while index < len(unique_pickup_points): # O(P) = O(1) time loop; loop P times (P<= 18)
        pickup_location = unique_pickup_points[index] # O(1) time: get unique pickup location ID
        distance = dijkstra_from(pickup_location) #O(R + L log L) time: run Dijkstra's from this lcoation
        bus_list = buses_at_pickup_point[index] #O(1) time: get list of bus IDs at this location
        
       # O(S) time loop: check reachability for all S students
        for student_id, student_loc in enumerate(students): 
            if distance[student_loc] <= D: # O(1) time: check if student can reach this location
                for bus_id in bus_list: # O(B) iterate buses at this point 
                    reachable_by_bus[bus_id].append(student_id) # O(1 ) time; add student to bus's list
                    buses_by_student[student_id].append(bus_id) # O(1) time; add bus to student's list 
        
        # O(1) time; increment index
        index += 1


     # Flow Network
    # O(1) time: define node indices
    NETWORK_SOURCE = 0
    NETWORK_SINK = 1
    STUDENT_NODE_START = 2
    BUS_NODE_START = STUDENT_NODE_START + S
    DEMAND_SUPER_SOURCE = BUS_NODE_START + B
    DEMAND_SUPER_SINK = DEMAND_SUPER_SOURCE + 1
    NODE_COUNT = DEMAND_SUPER_SINK + 1 # total nnodes V = S + b + 4

    DEST, REV_INDEX, CAPACITY = 0, 1, 2 # O(1) time: define edge tuple indices for clarity

    graph = [[] for _ in range(NODE_COUNT)] # O(V) = O(S +B) = O(S) aux space; initalsies flow graph

    def add_edge(u, v, cap_value):
        """
        Adds a bidirectional edge pair to represent a single directed edge
        in a residual graph for max-flow.
        
        1.  A "forward edge" (u->v) is added with the given capacity.
        2.  A "reverse edge" (v->u) is added with 0 capacity. This edge
            will hold "residual capacity" if flow is pushed along the
            forward edge.
            
        The `rev` index points to the reverse edge in the other node's
        adjacency list, allowing O(1) lookup to update the reverse
        edge's capacity when the forward edge is used.

        Time: O(1) 
          - `list.append()` is O(1) on average 
          - Worst-case: O(N) where N is the number of elements in the
            list, if the list's underlying array needs to be resized and copied. 
        """
        # forward edge
        graph[u].append([v, len(graph[v]), cap_value]) # O(1) time; append forward edge
        # backward edge
        graph[v].append([u, len(graph[u]) - 1, 0]) # O(1) time; append reverse edge

    def bfs_find_path(source, sink, limit):
        visited = [False] * NODE_COUNT # O(V) = O(S) time and space; intialise visted list
        parent_node = [-1] * NODE_COUNT # O(V) = O(S) time and space; store paretn node in the BFSpath
        parent_edge = [-1] * NODE_COUNT # O(V) = O(S) time and space; store edge  index used to reach node

        queue = [0] * NODE_COUNT # O(V) = O(S) aux space; intilaises a list based queue
        head = 0 # O(1) time; intilaises queue head pointer
        tail = 0 #O(1) time; intialise queue tail pointer
        queue[tail] = source   # O(1) time; Add source node to queue
        visited[source] = True # O(1) time; mark source as visited

        while head < tail: # O(V) = O(S) time loop; BFS explores each node at most once
            u = queue[head] # O(1) time; dequeue a node
            head += 1 # O(1) time; increment head
            adjacent_u = graph[u] #O(1) time; get adjaceny list for node u


            # iterate through all neighbours
            #O(E)= O(S) total for all loops
            j=0
            while j < len(adjacent_u):  
                e = adjacent_u[j] #O(1) time; edge
                if e[CAPACITY] > 0: #O(1) time; check if there is capacity
                    v = e[DEST] #O(1) time; check if visited
                    if not visited[v]: #O(1) time; desitnation node
                        visited[v] = True # O(1) time; mark visited
                        parent_node[v] = u #O(1) time; set parent
                        parent_edge[v] = j # O(1) time ; set edge index

                        if v == sink: #O(1) time; check if sink is reached
                            bottleneck = limit # O(1 time); set initial bottleneck
                            walk_node = sink # O(1) time; start walk back from sink
                            while walk_node != source: # O(V) = 0(S) time loop; walk back to source
                                prev_node = parent_node[walk_node] #O(1) time; gets parent node
                                prev_edge_index = parent_edge[walk_node] ##O(1) time; gets the edge index from parent
                                edge_capacity = graph[prev_node][prev_edge_index][CAPACITY] #O(1) time; get capacity of edge
                                if edge_capacity < bottleneck: #O(1) time; update bottleneck if this edge is smaller
                                    bottleneck = edge_capacity 
                                walk_node = prev_node # O(1) time; move to parent
                            return bottleneck, parent_node, parent_edge # O(1) time; return flow and path info
                        
                        queue[tail] = v #O(1) time; add node to queue
                        tail += 1 # O(1) time: increment tail
                j += 1 #O(1) time: go to next edge

        return 0, parent_node, parent_edge #O(1) time: no path found, retun 0 flow
    
    def maxflow(source, sink, flow_limit):

        total_flow = 0 # O(1) time; intilaises total flow

        while total_flow < flow_limit: # O(F) = O(flow_limit) time loop; find F augmenting paths
            remaining = flow_limit - total_flow # O(1) time; remainig flow needed is calcuated 
            pushed, parent_node, parent_edge = bfs_find_path(source, sink, remaining) # O(E) = O(S) time; find one augmenting path
            if pushed == 0: #O(1) time; stop if no path is found
                break
            
            node = sink #O(V) = O(S) time; augment path
            while node != source:
                prev_node = parent_node[node] #O(1) time; parent
                edge_index = parent_edge[node] # O(1) time; edge index is gotten from parent
                e = graph[prev_node][edge_index] # O(1) time; get forward edge
                e[CAPACITY] -= pushed #O(1) time subtract capacity from forward edge
                rev_edge_index = e[REV_INDEX] # O(1) time; reverse edge index
                graph[node][rev_edge_index][CAPACITY] += pushed #O(1) time; add capacity to reverse edge 
                node = prev_node #O(1) time; move to parent

            total_flow += pushed #O(1) time; add pushed flow to total

        return total_flow #O(1) time; return total flow pushed
    

    # SOURCE -> students
    # O(S) time loop
    student_id = 0
    while student_id < S:
        add_edge(NETWORK_SOURCE, STUDENT_NODE_START + student_id, 1) #O(1) time; add edge from source to student
        student_id += 1 #O(1) time; increment 

    
    # student -> bus
    #O(B) =O(1) time loop
    for bus_id in range(B):
        bus_node = BUS_NODE_START + bus_id # O(1) time; get bus node ID
        reachable_students = reachable_by_bus[bus_id] # O(1) time; list of studetns who can reach this bus
        for student_id in reachable_students: #O(S) time loop
            add_edge(STUDENT_NODE_START + student_id, bus_node, 1) # O(1) timel add edge from studetn to bus
        
    # bus -> sink 
    # O(S) aux space; list to store node demands
    demand = [0] * NODE_COUNT
    for bus_id in range(B): #O(B)= O(1) time loop
        lower_bound = min_caps[bus_id] #O(1) time loop; min capacities
        upper_bound = max_caps[bus_id] # O(1) time loop; max capacities
        bus_node = BUS_NODE_START + bus_id #O(1) time; get bus node ID

        add_edge(bus_node, NETWORK_SINK, upper_bound - lower_bound) # O(1); add edge from bus to sink with cap = (max - min)
        demand[bus_node] -= lower_bound # O(1) time; add negative demand to bus node 
        demand[NETWORK_SINK] += lower_bound # O(1) timel add psotive demand to sink node as it supplies flow

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
    
    
