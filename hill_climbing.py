"""
hill_climbing.py
----------------
Hill Climbing for parking slot allocation (steepest-descent version).
Start with a valid allocation, then repeatedly move to the best neighbouring
allocation (change ONE vehicle's slot) while the total cost decreases.
"""

import random
import time

from parking_model import ParkingLot, create_vehicles, print_layout


def initial_allocation(lot, vehicles, rng):
    """Build a valid starting allocation: each vehicle gets a random free compatible slot."""
    used = set()
    allocation = {}
    for v in vehicles:
        options = [s.slot_id for s in lot.get_compatible_vacant_slots(v)
                   if s.slot_id not in used]
        chosen = rng.choice(options)
        allocation[v.vehicle_id] = chosen
        used.add(chosen)
    return allocation


def are_adjacent(slot_a, slot_b):
    """True if two slots touch each other (up, down, left or right)."""
    return abs(slot_a.row - slot_b.row) + abs(slot_a.col - slot_b.col) == 1


def generate_neighbors(lot, vehicles, allocation, neighborhood="any"):
    """
    Neighbour = the same allocation with ONE vehicle moved to another slot.
    neighborhood = "any"      : the vehicle may move to any free compatible slot
                   "adjacent" : the vehicle may only move to a touching slot
    Constraints are checked here, so every neighbour is valid.
    """
    neighbors = []
    used = set(allocation.values())
    for v in vehicles:
        current = lot.slots[allocation[v.vehicle_id]]
        for s in lot.get_compatible_vacant_slots(v):
            if s.slot_id in used:                 # unique-slot constraint
                continue
            if neighborhood == "adjacent" and not are_adjacent(current, s):
                continue
            new_alloc = dict(allocation)
            new_alloc[v.vehicle_id] = s.slot_id
            neighbors.append(new_alloc)
    return neighbors


def hill_climbing_allocate(lot, vehicles, seed=None, start=None, neighborhood="any"):
    """
    Allocate slots using Hill Climbing. The lot is NOT modified.
    seed    : makes the random start repeatable
    start   : optional starting allocation (used for demonstrations)
    Returns the same dictionary format as bfs_allocate and astar_allocate.
    """
    start_time = time.perf_counter()
    rng = random.Random(seed)
    explored = 0
    allocation = None
    message = ""

    if not lot.can_allocate_all(vehicles):
        message = "No valid allocation: not enough vacant compatible slots"
    else:
        current = dict(start) if start else initial_allocation(lot, vehicles, rng)
        current_cost = lot.calculate_total_cost(current)
        explored += 1                              # the starting state

        while True:
            best_neighbor, best_cost = None, current_cost
            for neighbor in generate_neighbors(lot, vehicles, current, neighborhood):
                explored += 1                      # every neighbour we evaluate
                cost = lot.calculate_total_cost(neighbor)
                if cost < best_cost:               # strictly better only
                    best_neighbor, best_cost = neighbor, cost
            if best_neighbor is None:              # no better neighbour: stop
                break
            current, current_cost = best_neighbor, best_cost
        allocation = current

    time_ms = (time.perf_counter() - start_time) * 1000

    success = allocation is not None
    total_cost = None
    if success:
        ok, check_message = lot.validate_allocation(vehicles, allocation)
        if not ok:                                 # safety net, should never happen
            success, allocation, message = False, None, check_message
        else:
            total_cost = lot.calculate_total_cost(allocation)
            message = "Allocation found"
    elif not message:
        message = "No valid allocation found"

    return {
        "algorithm": "Hill Climbing",
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


# ---------- Self-tests (run only when this file is run directly) ----------
if __name__ == "__main__":
    from astar_algorithm import astar_allocate

    # Test 1
    lot = ParkingLot()
    lot.set_occupied_slots(["P2", "P7", "P12"])
    vehicles = create_vehicles(2, 1)
    result = hill_climbing_allocate(lot, vehicles, seed=1)
    show_result("Test 1: P2, P7, P12 occupied; C1, C2, B1 (seed 1)", result)
    print_layout(lot, result["allocation"])
    print()

    # Test 2: no compatible slot
    lot = ParkingLot()
    lot.set_occupied_slots(["P16", "P17", "P18", "P19", "P20"])
    show_result("Test 2: all bike slots occupied; B1",
                hill_climbing_allocate(lot, create_vehicles(0, 1), seed=1))

    # Test 3: no vehicles
    lot = ParkingLot()
    show_result("Test 3: no vehicles", hill_climbing_allocate(lot, [], seed=1))

    # Test 4: Hill Climbing vs A* over 20 random starts
    print("=== Test 4: Hill Climbing (any slot) vs A*, 20 random starts ===")
    lot = ParkingLot()
    lot.set_occupied_slots(["P2", "P7", "P12"])
    vehicles = create_vehicles(3, 2)
    best = astar_allocate(lot, vehicles)["total_cost"]
    costs = [hill_climbing_allocate(lot, vehicles, seed=s)["total_cost"] for s in range(20)]
    print("A* optimal cost      :", best)
    print("Hill Climbing costs  :", costs)
    print("Runs that hit optimum:", sum(1 for c in costs if c == best), "of 20")
    print()

    # Test 5: LOCAL OPTIMUM demonstration (neighbours = touching slots only)
    lot = ParkingLot()
    lot.set_occupied_slots(["P6"])
    vehicles = create_vehicles(1, 0)
    stuck = hill_climbing_allocate(lot, vehicles, start={"C1": "P11"}, neighborhood="adjacent")
    optimal = astar_allocate(lot, vehicles)
    show_result("Test 5: Hill Climbing stuck (adjacent moves only, start at P11)", stuck)
    show_result("Test 5: A* on the same problem", optimal)
