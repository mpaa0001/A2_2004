import heapq


def assign(L, roads, students, buses, D, T):
   
    adjacent_list = [[] for _ in range(L)] #adjaceny list built
    for u, v, w in roads:
        adjacent_list[u].append((v, w))
        adjacent_list[v].append((u, w))




    B = len(buses)  #numnber of buses
    S = len(students) #number of students
    infinity = float('inf') #infinity value for Dijkstra's algorithm


    reachable_students = [[] for _ in range(B)] #list of reachable students for each bus


    for bus_index in range(B):
        pickup_point, _, _ = buses[bus_index]
        distances = [infinity] * L
        distances[pickup_point] = 0
        priority_queue = [(0, pickup_point)]


#dijkstra's alogrithm,
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            if current_distance != distances[current_node]:
                continue
            for neighbor, weight in adjacent_list[current_node]:
                new_distance = current_distance + weight
                if new_distance < distances[neighbor] and new_distance <= D:
                    distances[neighbor] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor))


        for student_index, student_location in enumerate(students):
            if distances[student_location] <= D:
                reachable_students[bus_index].append(student_index)
       
   
    total_min = 0
    total_max = 0
    for bus in buses:
        total_min += bus[1]
        total_max += bus[2]


    if not (total_min <= T <= total_max):
        return None #imposttibel total students
   


    allocation = [-1] * S
    solution_found = [False]


    # pruing helper
    def count_unassigned_reachable(bus_j):
        count = 0
        for s in reachable_students[bus_j]:
            if allocation[s] == -1:
                count += 1
        return count
   
    def bounds_from(start_bus):
        min_left = 0
        max_left = 0
        for j in range(start_bus, B):
            min_cap, max_cap = buses[j][1], buses[j][2]
            min_left += min_cap
            available = count_unassigned_reachable(j)
            max_left += min(max_cap, available)
        return min_left, max_left


    def backtrack(bus_id, assigned_so_far):
        if bus_id == B:
            solution_found[0] = (assigned_so_far == T)
            return
       
        pickup_point, min_cap, max_cap = buses[bus_id]
        candidates = []
        for student_id in reachable_students[bus_id]:
            if allocation[student_id] == -1:
                candidates.append(student_id)


        if len(candidates) < min_cap:
            return
       
        min_left, max_left = bounds_from(bus_id)
        if assigned_so_far + min_left> T:
            return
        if assigned_so_far + max_left < T:
            return
       
        upper = min(max_cap, len(candidates))


        for group_size in range(min_cap, upper + 1):
            selected = candidates[:group_size]


            for student_id in selected:
                allocation[student_id] = bus_id


            backtrack(bus_id + 1, assigned_so_far + group_size)
            if solution_found[0]:
                return
           
            for student_id in selected:
                allocation[student_id] = -1




    backtrack(0, 0)
    return allocation if solution_found[0] else None






