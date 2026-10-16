"""Download public Showdown Pokemon type/move data ONCE for local Riffo decisions.

No calls are made during a game. Run on the machine hosting the agent while online.
No credentials are used or stored. Data source: play.pokemonshowdown.com/data/.
"""
from __future__ import annotations
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "https://play.pokemonshowdown.com/data/"
OUTPUT = Path(__file__).resolve().parents[1] / "agent" / "pokemon_reference.json"


def pokemon_id(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def load(name):
    url = SOURCE + name + ".json"
    request = urllib.request.Request(url, headers={"User-Agent":"Riffo-Agent/6.1 (tournament reference setup)"})
    with urllib.request.urlopen(request, timeout=18) as stream:
        return json.load(stream)


def main():
    dex, moves = load("pokedex"), load("moves")
    assert isinstance(dex, dict) and isinstance(moves, dict)
    assert len(dex) >= 600 and len(moves) >= 500, "Unexpectedly small Showdown reference data"
    species_data = {}
    for key, row in dex.items():
        if not isinstance(row, dict) or not isinstance(row.get("types"), list):
            continue
        stats = row.get("baseStats") or {}
        abilities = row.get("abilities") or {}
        species_data[pokemon_id(key)] = {
            "types": row["types"],
            "base_speed": stats.get("spe") if isinstance(stats, dict) else None,
            "possible_abilities": sorted(set(v for v in abilities.values() if isinstance(v, str))) if isinstance(abilities, dict) else [],
        }
    move_data = {}
    for key, row in moves.items():
        if not isinstance(row, dict) or not isinstance(row.get("type"), str):
            continue
        move_data[pokemon_id(key)] = {
            "type": row["type"], "power": row.get("basePower", 0),
            "accuracy": row.get("accuracy"), "priority": row.get("priority", 0),
            "category": row.get("category"),
        }
    assert "dragonite" in species_data and "woodhammer" in move_data
    assert set(map(str.lower, species_data["dragonite"]["types"])) == {"dragon", "flying"}
    assert move_data["woodhammer"]["type"].lower() == "grass"
    out = {"source": SOURCE, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
           "species":species_data, "moves":move_data}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    temp.replace(OUTPUT)
    print(f"Verified Pokemon Showdown data: {len(species_data)} species, {len(move_data)} moves")
    print(f"Saved locally to {OUTPUT}")
    print("Safe to share this public data. No API keys are included.")

if __name__ == "__main__":
    main()
