"""
Central configuration for the F1 data extraction pipeline.

All constants, paths, channel definitions, and padding strategy
are defined here for consistency across modules.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"

HYBRID_ERA_START = 2014
VALIDATION_START_YEAR = 2023

SESSION_TYPE = "R"

# Number of past races used for the rolling-window feature computation.
ROLLING_WINDOW = 10

STATUS_ID_MAP: dict[int, str] = {
    1: "Finished",
    2: "Disqualified",
    3: "Accident",
    4: "Collision",
    5: "Engine",
    6: "Gearbox",
    7: "Transmission",
    8: "Clutch",
    9: "Hydraulics",
    10: "Electrical",
    11: "+1 Lap",
    12: "+2 Laps",
    13: "+3 Laps",
    14: "+4 Laps",
    15: "+5 Laps",
    16: "+6 Laps",
    17: "+7 Laps",
    18: "+8 Laps",
    19: "+9 Laps",
    20: "Spun off",
    21: "Radiator",
    22: "Suspension",
    23: "Brakes",
    24: "Differential",
    25: "Overheating",
    26: "Mechanical",
    27: "Tyre",
    28: "Driver Seat",
    29: "Puncture",
    30: "Driveshaft",
    31: "Retired",
    32: "Fuel pressure",
    33: "Front wing",
    34: "Water pressure",
    35: "Refuelling",
    36: "Wheel",
    37: "Throttle",
    38: "Steering",
    39: "Technical",
    40: "Electronics",
    41: "Broken wing",
    42: "Heat shield fire",
    43: "Exhaust",
    44: "Oil leak",
    45: "Wheel nut",
    46: "Not classified",
    47: "Pneumatics",
    48: "Handling",
    49: "Rear wing",
    50: "Fire",
    51: "Wheel rim",
    52: "Water leak",
    53: "Fuel pump",
    54: "Track rod",
    55: "Oil pressure",
    56: "Engine fire",
    58: "Chassis",
    59: "Battery",
    60: "Socket",
    61: "Crankshaft",
    62: "Injection",
    63: "Distributor",
    64: "Turbo",
    65: "CV joint",
    66: "Water pump",
    67: "Fatal accident",
    68: "Spark plugs",
    69: "Fuel pipe",
    70: "Eye injury",
    71: "Oil pump",
    72: "Fuel rig",
    73: "Launch control",
    74: "Injured",
    75: "Fuel",
    76: "Power loss",
    77: "Vibrations",
    78: "107% Rule",
    79: "Safety",
    80: "Drivetrain",
    81: "Ignition",
    82: "Did not qualify",
    83: "Injury",
    84: "Undertray",
    85: "Debris",
    86: "Seat",
    87: "Damage",
    88: "Cooling system",
    89: "Withdrew",
    90: "Fuel system",
    91: "Tyre puncture",
    92: "Power Unit",
    93: "ERS",
    94: "Brake duct",
    95: "Seat belt",
    96: "Halfshaft",
    97: "Safety concerns",
    98: "Out of fuel",
    99: "Illness",
    100: "Excluded",
    101: "Underweight",
    102: "Alternator",
    103: "Physical",
    104: "Collision damage",
    105: "Mechanical problem",
    107: "Engine misfire",
    108: "Turbo charger",
    109: "Power supply",
    110: "Fuel leak",
    128: "Hose",
    129: "Fuel injection",
    130: "Did not prequalify",
    131: "Withdrew",
    132: "Not restarted",
    133: "Neck",
    134: "Finger",
    135: "Leg",
    136: "Back",
    137: "Concussion",
    138: "Fractured skull",
    139: "Internal organs",
    140: "Broken arm",
    141: "Loose wheel",
}

DRIVER_ERROR_STATUSES: list[str] = [
    "Accident",
    "Collision",
    "Collision damage",
    "Spun off",
]

FINISHED_STATUS_IDS: set[int] = {1, 11, 12, 13, 14, 15, 16, 17, 18, 19}

RELBENCH_DATASET = "rel-f1"
TASK_NAME = "driver-top3"
