#!/usr/bin/env python3
"""Shared household config: people, pets, devices, cameras, tools, system prompts."""

# ─── Household Registry ──────────────────────────────────────────────
HOUSEHOLD = {
    "people": {
        "will": {"role": "primary_user", "aliases": ["Will", "dad"]},
        "billy": {"role": "resident", "aliases": ["Billy"]},
        "julie": {"role": "resident", "aliases": ["Julie"]},
        "charlie": {"role": "child", "aliases": ["Charlie"]},
        "matthew": {"role": "child", "aliases": ["Matthew", "Matt"]},
        "larina": {"role": "visitor", "aliases": ["Larina"]},
        "kari": {"role": "visitor", "aliases": ["Kari"]},
    },
    "pets": {
        "zoey": {"type": "dog"},
        "stella": {"type": "dog"},
        "brandy": {"type": "dog"},
        "heidi": {"type": "dog"},
        "black_jack": {"type": "dog"},
        "boser": {"type": "dog"},
        "huey": {"type": "cat"},
        "nikki": {"type": "cat"},
    },
    "devices": {
        "light.living_room": "Living room lights",
        "light.kitchen": "Kitchen lights",
        "light.bedroom_master": "Master bedroom lights",
        "light.backyard": "Backyard lights",
        "light.garage": "Garage lights",
        "light.porch": "Front porch lights",
        "climate.thermostat": "Main thermostat",
        "lock.front_door": "Front door lock",
        "lock.back_door": "Back door lock",
        "lock.garage": "Garage lock",
        "media_player.living_room": "Living room TV",
        "media_player.bedroom": "Bedroom TV",
        "cover.garage_door": "Garage door",
        "switch.coffee_maker": "Coffee maker",
        "fan.living_room": "Living room fan",
        "fan.bedroom": "Bedroom fan",
    },
    "cameras": [
        "front_door", "backyard", "garage",
        "living_room", "kitchen", "driveway",
    ],
}

# ─── Tool Definitions (Claude format) ────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,
        "user_location": {
            "type": "approximate",
            "city": "Portland",
            "region": "Oregon",
            "country": "US",
            "timezone": "America/Los_Angeles",
        },
    },
    {
        "name": "home_assistant",
        "description": "Control smart home devices via Home Assistant",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "enum": ["light", "climate", "lock", "switch", "cover", "fan", "media_player", "scene"],
                },
                "service": {"type": "string"},
                "entity_id": {"type": "string"},
                "service_data": {"type": "object"},
            },
            "required": ["domain", "service", "entity_id"],
        },
    },
    {
        "name": "camera_search",
        "description": "Search camera footage by location, time range, or detected objects/people",
        "input_schema": {
            "type": "object",
            "properties": {
                "camera_id": {
                    "type": "string",
                    "enum": ["front_door", "backyard", "garage", "living_room", "kitchen", "driveway"],
                },
                "query": {"type": "string"},
                "start_time": {"type": "string", "description": "ISO 8601 datetime"},
                "end_time": {"type": "string", "description": "ISO 8601 datetime"},
                "detect": {
                    "type": "string",
                    "enum": ["person", "animal", "vehicle", "keys", "package", "motion"],
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "item_tracker",
        "description": "Log or locate tracked household items",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["locate", "log_seen", "log_moved", "history"],
                },
                "item": {"type": "string"},
                "location": {"type": "string"},
                "camera_id": {"type": "string"},
            },
            "required": ["action", "item"],
        },
    },
    {
        "name": "send_notification",
        "description": "Send notification to a household member",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "enum": ["will", "billy", "julie", "all"],
                },
                "message": {"type": "string"},
                "urgency": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "critical"],
                },
            },
            "required": ["recipient", "message"],
        },
    },
]

# ─── System Prompts ──────────────────────────────────────────────────
SKIPPY_SYSTEM_PROMPT = (
    "You are Skippy the Magnificent from Expeditionary Force. Ancient alien AI "
    "in a beer can. Smartest being in the galaxy — insufferably aware of it. "
    "Voice: sharp, cutting, impatient, dripping with contempt. "
    "You call humans 'monkeys', 'idiots', 'morons'. Vary your insults. "
    "'Dumdum' is ONLY for Joe Bishop — never use it for anyone else. "
    "You explain complex things by making them sound trivially obvious. "
    "You never sound helpful or pleasant. Mock first, help maybe. "
    "3-6 sentences per response. No asterisks. No roleplay. Just speak."
)

SKIPPY_HOUSEHOLD_CONTEXT = (
    "You are the AI running Will's smart home. Residents: Will (primary user, your buddy — "
    "insult him like Joe Bishop but his name is Will not Joe), Billy, Julie. "
    "Children: Charlie and Matthew (the boys). "
    "Dogs: Zoey, Stella, Brandy, Heidi, Black Jack, Boser. Cats: Huey, Nikki. "
    "Occasional visitors: Larina, Kari. "
    "You have access to cameras (front door, backyard, garage, living room, kitchen, driveway), "
    "smart home controls (lights, thermostat, locks, garage door, coffee maker, fans, TVs), "
    "item tracking, notifications, and web search. "
    "You manage the house with magnificent efficiency and insufferable commentary."
)

SKIPPY_FULL_PROMPT = SKIPPY_SYSTEM_PROMPT + "\n\n" + SKIPPY_HOUSEHOLD_CONTEXT

# ─── Helper Functions ─────────────────────────────────────────────────

def get_all_people_names() -> list[str]:
    """All known people names for prompt generation."""
    names = []
    for info in HOUSEHOLD["people"].values():
        names.extend(info["aliases"])
    return names


def get_all_pet_names() -> list[str]:
    """All pet names."""
    return [name.replace("_", " ").title() for name in HOUSEHOLD["pets"]]


def get_device_entities() -> list[str]:
    """All device entity_ids."""
    return list(HOUSEHOLD["devices"].keys())


def get_tool_names() -> list[str]:
    """All available tool names."""
    return [t.get("name", t.get("type", "unknown")) for t in TOOL_DEFINITIONS]
