"""Parse the non-input event details of a narrative into an event signature.

Every SafeTraffic Illinois narrative uses one template. Beyond the four input
conditions it states the weather, the road layout and traffic control (the
intersection context), whether the crash was in a work zone, the secondary
cause, and for each vehicle a model year, compass heading, and maneuver.

The event signature keeps the details that change the scenario:

- weather, road_layout, traffic_control, work_zone, secondary_cause;
- heading_relation: how the first two vehicles' headings relate (same,
  opposite, crossing, angled). Absolute compass directions are left out, since
  a north/south crash and an east/west crash are the same scenario;
- maneuvers: the distinct vehicle maneuvers, as a set so that the 3+ vehicle
  bin does not vary just by vehicle count. In the source data every vehicle is
  "traveling straight ahead", so this field only changes if a model invents
  other maneuvers.

Model years are left out as cosmetic.
"""

import re


WEATHER_RE = re.compile(r"lighting conditions and (.*?) weather at the time of the crash", re.S)
LAYOUT_RE = re.compile(r"The crash occurred on an? (.*?) with (.*?) in place, and it was confirmed "
                       r"that the crash (did not occur|happened|occurred) in a work zone", re.S)
SECONDARY_RE = re.compile(r"with secondary contributing cause \((.*?)\)", re.S)
STATED_COUNT_RE = re.compile(r"The crash involved (\d+) vehicles?", re.I)
VEHICLE_RE = re.compile(r"Vehicle (\d+), an? (.*?) model, was moving (\w+) and was (.*?) "
                        r"before the crash\.", re.S)
HEADINGS = {"North": 0, "Northeast": 45, "East": 90, "Southeast": 135, "South": 180,
            "Southwest": 225, "West": 270, "Northwest": 315}
SIGNATURE_FIELDS = ("weather", "road_layout", "traffic_control", "work_zone",
                    "secondary_cause", "heading_relation", "maneuvers")
# Source narratives with this many vehicles or more omit per-vehicle sentences.
NO_VEHICLE_SENTENCES_FROM = 5


def heading_relation(vehicles):
    if len(vehicles) < 2:
        return "single" if vehicles else "none"
    a, b = (HEADINGS.get(v["heading"]) for v in vehicles[:2])
    if a is None or b is None:
        return "unknown"
    diff = abs(a - b) % 360
    diff = min(diff, 360 - diff)
    return {0: "same", 180: "opposite", 90: "crossing"}.get(diff, "angled")


def parse_events(text):
    """Event details of one narrative; a field that cannot be found is None."""
    weather = WEATHER_RE.search(text)
    layout = LAYOUT_RE.search(text)
    secondary = SECONDARY_RE.search(text)
    stated = STATED_COUNT_RE.search(text)
    vehicles = [{"index": int(i), "year": year.strip(), "heading": heading,
                 "maneuver": maneuver.strip()}
                for i, year, heading, maneuver in VEHICLE_RE.findall(text)]
    events = {
        "weather": weather.group(1).strip() if weather else None,
        "road_layout": layout.group(1).strip() if layout else None,
        "traffic_control": layout.group(2).strip() if layout else None,
        "work_zone": (layout.group(3) != "did not occur") if layout else None,
        "secondary_cause": secondary.group(1).strip() if secondary else None,
        "heading_relation": heading_relation(vehicles),
        "maneuvers": tuple(sorted({v["maneuver"] for v in vehicles})),
        "vehicles": vehicles,
        "stated_vehicle_count": int(stated.group(1)) if stated else None,
    }
    events["parsed"] = all(events[f] is not None for f in SIGNATURE_FIELDS)
    events["vehicle_count_consistent"] = vehicle_count_consistent(events)
    return events


def vehicle_count_consistent(events):
    """Do the vehicle sentences agree with the narrative's own stated count?

    Matches the source convention: one sentence per vehicle, numbered from 0,
    and none at all for crashes with five or more vehicles.
    """
    stated, vehicles = events["stated_vehicle_count"], events["vehicles"]
    if stated is None:
        return None
    if stated >= NO_VEHICLE_SENTENCES_FROM and not vehicles:
        return True
    return [v["index"] for v in vehicles] == list(range(stated))


def signature(events):
    """Hashable event signature, or None if any of its fields is missing."""
    if not events["parsed"]:
        return None
    return tuple(events[f] for f in SIGNATURE_FIELDS)
