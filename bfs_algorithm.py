"""
bfs_algorithm.py
----------------
Breadth-First Search for parking slot allocation.
Returns the FIRST complete valid allocation it finds (not necessarily the cheapest).
"""

import time
from collections import deque

from parking_model import ParkingLot, create_vehicles, print_layout

MAX_STATES = 500_000   # safety limit so BFS cannot run forever on big problems


def bfs_allocate(lot, vehicles, max_states=MAX_STATES):
    """
    Allocate slots to vehicles using BFS.

    lot      : ParkingLot (it is NOT modified)
    vehicles : list of Vehicle
    Returns a dictionary with the same keys for every algorithm:
        algorithm, allocation, total_cost, explored_states,
        time_ms, success, message
    """
    start_time = time.perf_counter()
    explored = 0
    allocation = None
    message = ""

    if not lot.can_allocate_all(vehicles):
        # Constraint: vehicles cannot exceed available compatible slots
        message = "No valid allocation: not enough vacant compatible slots"
    else:
        # For each vehicle, the compatible vacant slots in slot order
        candidates = [
            [s.slot_id for s in lot.get_compatible_vacant_slots(v)]
            for v in vehicles
        ]

        queue = deque()
        queue.append(())          # start state: nobody assigned yet

        while queue:
            state = queue.popleft()       # FIFO = breadth-first
            explored += 1

            # Goal check: every vehicle has a slot
            if len(state) == len(vehicles):
                allocation = {v.vehicle_id: sid for v, sid in zip(vehicles, state)}
                break

            if explored >= max_states:
                message = f"Stopped: reached the limit of {max_states} states"
                break

            # Successor generation: next vehicle tries each slot not yet used
            next_index = len(state)
            for slot_id in candidates[next_index]:
                if slot_id not in state:          # a slot cannot be used twice
                    queue.append(state + (slot_id,))

    time_ms = (time.perf_counter() - start_time) * 1000

    success = allocation is not None
    total_cost = None
    if success:
        ok, check_message = lot.validate_allocation(vehicles, allocation)
        if not ok:                                # safety net, should never happen
            success, allocation, message = False, None, check_message
        else:
            total_cost = lot.calculate_total_cost(allocation)
            message = "Allocation found"
    elif not message:
        message = "No valid allocation found"

    return {
        "algorithm": "BFS",
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
    # Test 1: some slots occupied, 2 cars + 1 bike
    lot = ParkingLot()
    lot.set_occupied_slots(["P2", "P7", "P12"])
    vehicles = create_vehicles(2, 1)
    result = bfs_allocate(lot, vehicles)
    show_result("Test 1: P2, P7, P12 occupied; C1, C2, B1", result)
    print_layout(lot, result["allocation"])
    print()

    # Test 2: BFS is NOT optimal. P5 (cost 4) is tried before P6 (cost 1).
    lot = ParkingLot()
    lot.set_occupied_slots(["P1", "P2", "P3", "P4"])
    result = bfs_allocate(lot, create_vehicles(2, 0))
    show_result("Test 2: P1-P4 occupied; C1, C2", result)

    # Test 3: no compatible slot (all 5 bike slots occupied, 1 bike arrives)
    lot = ParkingLot()
    lot.set_occupied_slots(["P16", "P17", "P18", "P19", "P20"])
    result = bfs_allocate(lot, create_vehicles(0, 1))
    show_result("Test 3: all bike slots occupied; B1", result)

    # Test 4: no vehicles waiting
    lot = ParkingLot()
    result = bfs_allocate(lot, [])
    show_result("Test 4: no vehicles", result)
    
