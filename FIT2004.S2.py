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



    total_min_required = 0
    max_allowed = 0
    for _, min_cap, max_cap in buses:
        total_min_required += min_cap
        max_allowed += max_cap
    if T < total_min_required or T > max_allowed:
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
    def try_assign_student(student_id, visited_buses, visited_students, cap, min_quota, target_bus_id):
        for bus_position in range(len(buses_by_student[student_id])):
            bus_id = buses_by_student[student_id][bus_position]
            if visited_buses[bus_id]:
                continue
            visited_buses[bus_id] = True

            # direct assignment if bus has capacity
            if curr_no_students_bus[bus_id] < cap[bus_id]:
                add_to_bus(student_id, bus_id)
                return True
            
            # attempt to reassign students from this bus
            if curr_no_students_bus[bus_id] > min_quota[bus_id] or bus_id == target_bus_id:
                occupants = assigned_list[bus_id]
                for index in range(len(occupants)):
                    occupant_id = occupants[index]

                    if allocation[occupant_id] != bus_id or visited_students[occupant_id]:
                        continue
                    visited_students[occupant_id] = True


                    
                    
                    #temp remvoe occupant and try place them somewhere else
                    remove_from_bus(occupant_id, bus_id)
                    if try_assign_student(occupant_id, visited_buses, visited_students, cap, min_quota, target_bus_id):
                        add_to_bus(student_id, bus_id)
                        return True
                    add_to_bus(occupant_id, bus_id) #backtrack

        return False
    
    # enfore minimal capacities
    min_caps = []
    max_caps = []
    for pickup, min_cap, max_cap in buses:
        min_caps.append(min_cap)
        max_caps.append(max_cap) 

    cap = min_caps[:]
    min_quota_zero = [0] * B
    for student_id in range(S):
        if len(buses_by_student[student_id]) == 0:
            continue

        visited_buses = [False] * B
        visited_students = [False] * S
        try_assign_student(student_id, visited_buses, visited_students, cap, min_quota_zero, -1)


    # check all minima met
    for bus_id in range(B):
        if curr_no_students_bus[bus_id] < min_caps[bus_id]:
            return None #not enough students assigned to bus
        

    # add exaclty T - total_min_required students
    extra_needed = T - total_min_required
    if extra_needed == 0:
        return allocation
    
    cap = max_caps[:]
    min_quota = min_caps[:] 

    def try_add_one(extra_unassigned_only):
        for student_id in range(S):
            if extra_unassigned_only and allocation[student_id] != -1:
                continue

            targets = buses_by_student[student_id]
            for t in range(len(targets)):
                target_bus_id = targets[t]
                visited_buses = [False] * B
                visited_students = [False] * S

                if allocation[student_id] != -1:
                    old_bus = allocation[student_id]
                    remove_from_bus(student_id, old_bus)
                    was_assigned = try_assign_student(student_id, visited_buses, visited_students, cap, min_quota, target_bus_id)

                    if was_assigned:
                        return True
                    add_to_bus(student_id, old_bus) #backtrack
                else:
                    if try_assign_student(student_id, visited_buses, visited_students, cap, min_quota, target_bus_id):
                        return True
        return False
    
    extra_needed = T - total_min_required
    if extra_needed == 0:
        return allocation
    
    added = 0
    for only_unassigned in [True, False]:
        while added < extra_needed:
            if not try_add_one(only_unassigned):
                break
            added += 1

        if added == extra_needed:
            break

    if added != extra_needed:
        return None #unable to assign enough students
    
    return allocation

        
          

        
# ---------------------------
# TEST HARNESS (put this after your assign function)
# ---------------------------
import heapq

def run_test_case(name, L, roads, students, buses, D, T):
    print("\n" + "="*60)
    print(name)
    print("="*60)

    allocation = assign(L, roads, students, buses, D, T)
    print("Allocation:", allocation)

    if allocation is None:
        print("Result: None (no valid assignment)")
        return

    # --- Validate constraints ---
    S = len(students)
    B = len(buses)

    # Build adjacency once
    adj = [[] for _ in range(L)]
    for u, v, w in roads:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Dijkstra helper (no early D pruning needed for validation)
    INF = float('inf')
    def dijkstra(src):
        dist = [INF] * L
        final = [False] * L
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if final[u]:
                continue
            final[u] = True
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # Precompute distances from each bus pickup
    bus_dists = [dijkstra(buses[b][0]) for b in range(B)]

    # Check per-student reachability and count loads
    loads = [0] * B
    assigned_count = 0
    ok = True

    for i in range(S):
        b = allocation[i]
        if b == -1:
            continue
        assigned_count += 1
        loads[b] += 1
        loc = students[i]
        if bus_dists[b][loc] > D:
            print(f"Violation: student {i} at loc {loc} too far for bus {b} (dist={bus_dists[b][loc]}, D={D})")
            ok = False

    # Check exact T
    if assigned_count != T:
        print(f"Violation: total assigned {assigned_count} != T {T}")
        ok = False

    # Check per-bus min/max
    for b in range(B):
        mn, mx = buses[b][1], buses[b][2]
        if loads[b] < mn or loads[b] > mx:
            print(f"Violation: bus {b} load {loads[b]} not in [{mn}, {mx}]")
            ok = False

    print("Bus loads:", loads)
    print("VALID ✅" if ok else "FAILED ❌")


# ---------------------------
# CASE 1 (from the spec)
# ---------------------------
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
    (0, 3, 5),    # Bus 0
    (6, 5, 10),   # Bus 1
    (15, 5, 10),  # Bus 2
    (6, 5, 10)    # Bus 3
]
D = 5
T = 22

run_test_case("CASE 1", L, roads, students, buses, D, T)

# ---------------------------
# CASE 2 (from the spec)
# ---------------------------
students2 = [5, 8, 3, 7, 7, 15, 15, 8, 15, 7, 6, 15]
buses2 = [(0, 3, 5), (15, 5, 6)]
D2 = 5
T2 = 7

run_test_case("CASE 2", L, roads, students2, buses2, D2, T2)


    
                    


           




