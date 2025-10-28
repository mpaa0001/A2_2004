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
        pickup_point, min_cap, max_cap = buses[bus_index]
        distances = [infinity] * L
        distances[pickup_point] = 0
        priority_queue = [(0, pickup_point)]

        