import heapq

def assign(L, roads, students, buses, D, T):
    
    adjacent_list = [[] for _ in range(L)]
    for u, v, w in roads:
        adjacent_list[u].append((v, w))
        adjacent_list[v].append((u, w))

    B = len(buses)
    S = len(students)

    unique_pickup_points = []
    buses_at_pickup_point = []
    pickup_to_index = [-1] * L

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
    
    infinity = max(total_max + 1, D _ 1)
       


    