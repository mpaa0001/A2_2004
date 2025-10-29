import heapq


def assign(L, roads, students, buses, D, T):
   
    adjacent_list = [[] for _ in range(L)] #adjaceny list built
    for u, v, w in roads:
        adjacent_list[u].append((v, w))
        adjacent_list[v].append((u, w))




    B = len(buses)  #numnber of buses
    S = len(students) #number of students
    infinity = float('inf') #infinity value for Dijkstra's algorithm


    reachable_by_bus= [[] for _ in range(B)] #list of reachable students for each bus
    for bus_id in range(B):
        pickup, _, _ = buses[bus_id]
        distances = [infinity] * L
        finalised = [False] * L
        distances[pickup] = 0
        priority_queue = [(0, pickup)]

        while priority_queue:
            current_distance, u = heapq.heappop(priority_queue)
            if finalised[u]:
                continue
            if current_distance > D:
                break
            finalised[u] = True

            for v, w in adjacent_list[u]:
                new_distance = current_distance + w
                if new_distance <= D and new_distance < distances[v]:
                    distances[v] = new_distance
                    heapq.heappush(priority_queue, (new_distance, v))

        for student_id, location_id in enumerate(students):
            if distances[location_id] <= D:
                reachable_by_bus[bus_id].append(student_id)

    # for each student, checks which buses can reach them
    buses_by_student = [[] for _ in range(S)] #list of buses that can reach each student
    for bus_id, student_list in enumerate(reachable_by_bus):
        for student_id in student_list:
            buses_by_student[student_id].append(bus_id)  



    min_required = 0
    max_allowed = 0
    for _, min_cap, max_cap in buses:
        min_required += min_cap
        max_allowed += max_cap
    if T < min_required or T > max_allowed:
        return None #impossible total students
    

    allocation = [-1] * S
    curr_no_students_bus = [0] * B
    assigned_list = [[] for _ in range(B)] 

    def add_to_bus(student_id, bus_id):
        allocation[student_id] = bus_id
        curr_no_students_bus[bus_id] += 1
        assigned_list[bus_id].append(student_id)

    def remove_from_bus(student_id, bus_id):
        allocation[student_id] = -1
        curr_no_students_bus[bus_id] -= 1

    # dfs method
    def try_assign_student(student_id, visited_buses, visited_students, cap, floor_cap, start_target):
        for bus_position in range(len(buses_by_student[student_id])):
            bus_id = buses_by_student[student_id][bus_position]
            if visited_buses[bus_id]:
                continue
            visited_buses[bus_id] = True

            if curr_no_students_bus[bus_id] < cap[bus_id]:
                add_to_bus(student_id, bus_id)
                return True
            
            if curr_no_students_bus[bus_id] > floor_cap[bus_id] or bus_id == start_target:
                
            


           








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






