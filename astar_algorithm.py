"""
astar_algorithm.py
------------------
A* Search for parking slot allocation.
f(n) = g(n) + h(n)
  g(n): total distance cost of the slots assigned so far
  h(n): sum of the cheapest possible costs for the vehicles still waiting
        (admissible: it never overestimates the real remaining cost)
"""

import heapq
import time

from parking_model import ParkingLot, create_vehicles, print_layout
from bfs_algorithm import bfs_allocate

MAX_STATES = 500_000   # safety limit


def heuristic(state, vehicles, slots_by_type):
    """
    h(n): lower bound on the remaining cost.
    For each vehicle type, the waiting vehicles need k different slots,
    so the cost is at least the sum of the k cheapest unused slots of that type.
    """
    used = set(state)
    remaining = vehicles[len(state):]
    total = 0
    for vtype in set(v.vehicle_type for v in remaining):
        k = sum(1 for v in remaining if v.vehicle_type == vtype)
        costs = [c for (c, sid) in slots_by_type.get(vtype, []) if sid not in used]
        total += sum(costs[:k])        # slots_by_type is sorted, so these are the cheapest
    return total


def astar_allocate(lot, vehicles, max_states=MAX_STATES):
    """
    Allocate slots using A*. The lot is NOT modified.
    Returns the same dictionary format as bfs_allocate.
    """
    start_time = time.perf_counter()
    explored = 0
    allocation = None
    message = ""

    if not lot.can_allocate_all(vehicles):
        message = "No valid allocation: not enough vacant compatible slots"
    else:
        # Vacant slots grouped by type, sorted by cost: {"car": [(0,"P1"), ...], ...}
        slots_by_type = {}
        for s in lot.get_vacant_slots():
            slots_by_type.setdefault(s.slot_type, []).append((s.distance_cost, s.slot_id))
        for vtype in slots_by_type:
            slots_by_type[vtype].sort()

        candidates = [
            [s.slot_id for s in lot.get_compatible_vacant_slots(v)]
            for v in vehicles
        ]
        cost_of = {s.slot_id: s.distance_cost for s in lot.slots.values()}

        counter = 0                                   # tie-breaker for equal priorities
        start = ()
        h0 = heuristic(start, vehicles, slots_by_type)
        # Queue entries: (f, h, counter, state, g)
        frontier = [(h0, h0, counter, start, 0)]
        closed = set()                                # states already expanded

        while frontier:
            f, h, _, state, g = heapq.heappop(frontier)   # lowest f first

            if state in closed:                       # repeated state: skip
                continue
            closed.add(state)
            explored += 1

            # Goal check: every vehicle has a slot
            if len(state) == len(vehicles):
                allocation = {v.vehicle_id: sid for v, sid in zip(vehicles, state)}
                break

            if explored >= max_states:
                message = f"Stopped: reached the limit of {max_states} states"
                break

            # Successor generation
            for slot_id in candidates[len(state)]:
                if slot_id in state:                  # a slot cannot be used twice
                    continue
                new_state = state + (slot_id,)
                if new_state in closed:
                    continue
                new_g = g + cost_of[slot_id]
                new_h = heuristic(new_state, vehicles, slots_by_type)
                counter += 1
                heapq.heappush(frontier, (new_g + new_h, new_h, counter, new_state, new_g))

    time_ms = (time.perf_counter() - start_time) * 1000

    success = allocation is not None
    total_cost = None
    if success:
        ok, check_message = lot.validate_allocation(vehicles, allocation)
        if not ok:                                    # safety net, should never happen
            success, allocation, message = False, None, check_message
        else:
            total_cost = lot.calculate_total_cost(allocation)
            message = "Allocation found"
    elif not message:
        message = "No valid allocation found"

    return {
        "algorithm": "A*",
        "allocation": allocation,
        "total_cost": total_cost,
        "explored_states": explored,
        "time_ms": time_ms,
        "success": success,
        "message": message,
    }


def show_result(title, result):
    print(f"--- {title} ---")
    print("Success        :", result["success"])
    print("Message        :", result["message"])
    print("Allocation     :", result["allocation"])
    print("Total cost     :", result["total_cost"])
    print("Explored states:", result["explored_states"])
    print(f"Time (ms)      : {result['time_ms']:.4f}")
    print()


def brute_force_best_cost(lot, vehicles):
    """Try EVERY possible allocation and return the lowest cost (for checking A*)."""
    from itertools import permutations
    slot_ids = [s.slot_id for s in lot.get_vacant_slots()]
    best = None
    for combo in permutations(slot_ids, len(vehicles)):
        allocation = {v.vehicle_id: sid for v, sid in zip(vehicles, combo)}
        ok, _ = lot.validate_allocation(vehicles, allocation)
        if ok:
            cost = lot.calculate_total_cost(allocation)
            if best is None or cost < best:
                best = cost
    return best


# ---------- Self-tests (run only when this file is run directly) ----------
if __name__ == "__main__":
    # Test 1
    lot = ParkingLot()
    lot.set_occupied_slots(["P2", "P7", "P12"])
    vehicles = create_vehicles(2, 1)
    result = astar_allocate(lot, vehicles)
    show_result("Test 1: P2, P7, P12 occupied; C1, C2, B1", result)
    print_layout(lot, result["allocation"])
    print()

    # Test 2: A* beats BFS
    lot = ParkingLot()
    lot.set_occupied_slots(["P1", "P2", "P3", "P4"])
    vehicles = create_vehicles(2, 0)
    show_result("Test 2: A*", astar_allocate(lot, vehicles))
    show_result("Test 2: BFS (for comparison)", bfs_allocate(lot, vehicles))

    # Test 3: no compatible slot
    lot = ParkingLot()
    lot.set_occupied_slots(["P16", "P17", "P18", "P19", "P20"])
    show_result("Test 3: all bike slots occupied; B1", astar_allocate(lot, create_vehicles(0, 1)))

    # Test 4: no vehicles
    lot = ParkingLot()
    show_result("Test 4: no vehicles", astar_allocate(lot, []))

    # Test 5: A* cost must equal the brute-force optimum
    print("=== Optimality check: A* cost vs brute force ===")
    import random
    random.seed(1)
    all_ids = [f"P{i}" for i in range(1, 21)]
    all_match = True
    for trial in range(1, 11):
        lot = ParkingLot()
        lot.set_occupied_slots(random.sample(all_ids, random.randint(0, 8)))
        vehicles = create_vehicles(random.randint(0, 2), random.randint(0, 1))
        a = astar_allocate(lot, vehicles)
        best = brute_force_best_cost(lot, vehicles)
        same = (a["total_cost"] == best)
        all_match = all_match and same
        print(f"Trial {trial:>2}: A* cost = {a['total_cost']}, brute force = {best}, match = {same}")
    print("All trials match:", all_match)
