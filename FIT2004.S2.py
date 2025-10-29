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
        
        for bus_index, reach in enumerate(reachable_students):
            print(f"Bus {bus_index} can reach students: {reach}")

    total_min = 0
    total_max = 0 
    for bus in buses:
        total_min += bus[1]
        total_max += bus[2]

    if not (total_min <= T <= total_max):
        print("Impossible: total_min= ", total_min, "total_max= ", total_max)
        return None #imposttibel total students
    

    allocation = [-1] * S
    solution_found = [False]

    def backtrack(bus_id, assigned_so_far):
        if bus_id == B:
            solution_found[0] = (assigned_so_far == T)
            return
        
        pickup_point, min_cap, max_cap = buses[bus_id]
        candidates = []
        for student_id in reachable_students[bus_id]:
            if allocation[student_id] == -1:
                candidates.append(student_id)

        if assigned_so_far > T:
            return
        if assigned_so_far + len(candidates) < T:
            return
        
        for group_size in range(min_cap, max_cap + 1):
            if group_size > len(candidates):
                break
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




# TEST 1:

L = 16
roads = [
    (0,1,3), (0,2,5), (0,3,10), (1,4,1), (2,5,2), (5,6,3),
    (2,7,4), (0,8,1), (0,9,1), (0,10,1), (0,11,1), (6,12,2),
    (6,13,4), (6,14,3), (7,15,1)
]

students = [
    4, 10, 8, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
    5, 7, 7, 7, 7, 7, 15, 15, 7, 4, 8, 9
]

buses = [
    (0, 3, 5),   # Bus 0
    (6, 5, 10),  # Bus 1
    (15, 5, 10), # Bus 2
    (6, 5, 10)   # Bus 3
]

D = 5
T = 22

result1 = assign(L, roads, students, buses, D, T)
print("Test 1 Result:", result1)
