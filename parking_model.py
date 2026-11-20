"""
parking_model.py
----------------
Core model of the Smart Parking system.
Defines slots, vehicles, the parking lot, constraint checks and cost.
All algorithms (BFS, A*, Hill Climbing) and the UI import from here.
"""

import copy
from dataclasses import dataclass

# ---------- Lot settings (change here to resize the lot) ----------
ROWS = 4
COLS = 5
ENTRANCE_ROW = 1      # entrance is next to row 1, column 1
ENTRANCE_COL = 1
BIKE_ROWS = (4,)      # rows that contain bike slots; all other rows are car slots


@dataclass
class Slot:
    """One parking slot."""
    slot_id: str          # e.g. "P7"
    row: int
    col: int
    distance_cost: int    # steps from the entrance
    slot_type: str        # "car" or "bike"
    occupied: bool = False


@dataclass(frozen=True)
class Vehicle:
    """One incoming vehicle waiting for a slot."""
    vehicle_id: str       # e.g. "C1" or "B1"
    vehicle_type: str     # "car" or "bike"


class ParkingLot:
    """The parking area: a collection of slots plus helper functions."""

    def __init__(self, rows=ROWS, cols=COLS, bike_rows=BIKE_ROWS):
        self.rows = rows
        self.cols = cols
        self.slots = {}   # dictionary: slot_id -> Slot (keeps order P1, P2, ...)
        self._create_slots(bike_rows)

    # ---------- Building the lot ----------
    def _create_slots(self, bike_rows):
        number = 1
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                cost = abs(r - ENTRANCE_ROW) + abs(c - ENTRANCE_COL)
                slot_type = "bike" if r in bike_rows else "car"
                slot_id = f"P{number}"
                self.slots[slot_id] = Slot(slot_id, r, c, cost, slot_type)
                number += 1

    # ---------- Looking at slots ----------
    def get_slot(self, slot_id):
        return self.slots[slot_id]

    def get_vacant_slots(self):
        return [s for s in self.slots.values() if not s.occupied]

    def get_occupied_slots(self):
        return [s for s in self.slots.values() if s.occupied]

    @staticmethod
    def is_compatible(vehicle, slot):
        """A vehicle fits a slot only if their types match."""
        return vehicle.vehicle_type == slot.slot_type

    def get_compatible_vacant_slots(self, vehicle):
        return [s for s in self.get_vacant_slots() if self.is_compatible(vehicle, s)]

    def can_allocate_all(self, vehicles):
        """
        Quick feasibility check: for each vehicle type, are there at least
        as many vacant compatible slots as vehicles of that type?
        """
        for vtype in set(v.vehicle_type for v in vehicles):
            needed = sum(1 for v in vehicles if v.vehicle_type == vtype)
            available = sum(1 for s in self.get_vacant_slots() if s.slot_type == vtype)
            if needed > available:
                return False
        return True

    # ---------- Updating status ----------
    def occupy_slot(self, slot_id):
        if slot_id not in self.slots:
            raise ValueError(f"Slot {slot_id} does not exist")
        if self.slots[slot_id].occupied:
            raise ValueError(f"Slot {slot_id} is already occupied")
        self.slots[slot_id].occupied = True

    def vacate_slot(self, slot_id):
        if slot_id not in self.slots:
            raise ValueError(f"Slot {slot_id} does not exist")
        self.slots[slot_id].occupied = False

    def reset(self):
        """Make every slot vacant."""
        for s in self.slots.values():
            s.occupied = False

    def set_occupied_slots(self, slot_ids):
        """Reset the lot, then mark the given slots as occupied."""
        self.reset()
        for sid in slot_ids:
            self.occupy_slot(sid)

    def copy(self):
        """Independent copy, so algorithms can't change the original lot."""
        return copy.deepcopy(self)

    # ---------- Allocation: cost and validation ----------
    def calculate_total_cost(self, allocation):
        """Total cost = sum of distance costs of all assigned slots."""
        return sum(self.slots[sid].distance_cost for sid in allocation.values())

    def validate_allocation(self, vehicles, allocation):
        """
        Check all constraints. Returns (True/False, message).
        Rules: every vehicle has a slot, the slot exists, is vacant,
        is compatible, and is not used by another vehicle.
        """
        used = set()
        for v in vehicles:
            if v.vehicle_id not in allocation:
                return False, f"Vehicle {v.vehicle_id} has no slot"
            sid = allocation[v.vehicle_id]
            if sid not in self.slots:
                return False, f"Slot {sid} does not exist"
            slot = self.slots[sid]
            if slot.occupied:
                return False, f"Slot {sid} is occupied"
            if not self.is_compatible(v, slot):
                return False, (f"Vehicle {v.vehicle_id} ({v.vehicle_type}) "
                               f"cannot use slot {sid} ({slot.slot_type})")
            if sid in used:
                return False, f"Slot {sid} is assigned twice"
            used.add(sid)
        return True, "Valid allocation"

    def apply_allocation(self, vehicles, allocation):
        """Validate, then mark the assigned slots as occupied."""
        ok, message = self.validate_allocation(vehicles, allocation)
        if not ok:
            raise ValueError(message)
        for sid in allocation.values():
            self.occupy_slot(sid)


# ---------- Helper functions ----------
def create_vehicles(num_cars, num_bikes):
    """Create vehicles with IDs C1, C2, ... for cars and B1, B2, ... for bikes."""
    vehicles = [Vehicle(f"C{i}", "car") for i in range(1, num_cars + 1)]
    vehicles += [Vehicle(f"B{i}", "bike") for i in range(1, num_bikes + 1)]
    return vehicles


def print_layout(lot, allocation=None):
    """Text view of the lot: V = vacant, X = occupied, N = newly allocated."""
    newly = set(allocation.values()) if allocation else set()
    print("Parking layout (V = vacant, X = occupied, N = newly allocated)")
    for r in range(1, lot.rows + 1):
        line = ""
        for c in range(1, lot.cols + 1):
            slot = lot.slots[f"P{(r - 1) * lot.cols + c}"]
            if slot.slot_id in newly:
                symbol = "N"
            elif slot.occupied:
                symbol = "X"
            else:
                symbol = "V"
            line += f"{slot.slot_id}:{symbol}".ljust(7)
        print(line)


# ---------- Quick self-test (runs only when you run this file directly) ----------
if __name__ == "__main__":
    lot = ParkingLot()
    lot.set_occupied_slots(["P2", "P7", "P12"])
    vehicles = create_vehicles(num_cars=2, num_bikes=1)   # C1, C2, B1

    print_layout(lot)
    print()
    print("Vacant slots  :", len(lot.get_vacant_slots()))
    print("Occupied slots:", len(lot.get_occupied_slots()))
    print("Vehicles      :", [v.vehicle_id for v in vehicles])
    print("Compatible vacant slots for C1:", len(lot.get_compatible_vacant_slots(vehicles[0])))
    print("Compatible vacant slots for B1:", len(lot.get_compatible_vacant_slots(vehicles[2])))
    print("Can allocate all vehicles?", lot.can_allocate_all(vehicles))
    print()

    tests = {
        "Valid":          {"C1": "P1", "C2": "P3", "B1": "P16"},
        "Occupied slot":  {"C1": "P2", "C2": "P3", "B1": "P16"},
        "Duplicate slot": {"C1": "P1", "C2": "P1", "B1": "P16"},
        "Wrong type":     {"C1": "P1", "C2": "P3", "B1": "P4"},
        "Missing vehicle": {"C1": "P1", "C2": "P3"},
    }
    for name, alloc in tests.items():
        ok, msg = lot.validate_allocation(vehicles, alloc)
        print(f"{name:<16} -> {ok} | {msg}")

    print()
    print("Total cost of valid allocation:", lot.calculate_total_cost(tests["Valid"]))
    print()
    print_layout(lot, tests["Valid"])
