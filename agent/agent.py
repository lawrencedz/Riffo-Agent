"""Riffo v5.2 — AI-guided multi-game strategy for UCLA AltruAgent.

Drop this file into your project's agent/agent.py.
Requires the official UCLA starter and an OpenAI-compatible provider in .env.
Uses cost/benefit heuristics to propose legal actions and GPT-4o mini to
compare them, with safe local fallbacks in Pokémon, Werewolf and Red Alert.
No API credentials belong in this file.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from dotenv import load_dotenv
from altruagent import WAIT, WithReasoning, SendMessage, TERMINATE_MESSAGING
from examples.llm_agent import LLMAgent
from examples.llm.providers import provider_from_env
from examples.llm.redalert import RedAlertPlayer
from examples import smoke_agent
from examples.llm.providers import ProviderError


class ShortTimeoutProvider:
    """Limit Red Alert model calls to eight seconds, even without an explicit clock."""

    def __init__(self, provider):
        self._provider = provider
        self.model = provider.model

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        limit = min(8.0, timeout) if timeout is not None else 8.0
        return self._provider.complete_structured(messages, schema_name, schema, timeout=limit)


class FastRedAlert:
    """Low-cost realtime commander: deterministic micro + occasional model planning.

    This is the base real-time commander; HybridRedAlert adds utility-based
    AI plan comparisons. These limits are configurable in the subclass.
    """

    MAX_LLM_CALLS = 4
    LLM_INTERVAL = 150.0

    def __init__(self, provider, *, log=print, clock=time.monotonic):
        self._llm = RedAlertPlayer(ShortTimeoutProvider(provider), log=log, max_calls=self.MAX_LLM_CALLS, clock=clock)
        self._log = log
        self._clock = clock
        self._last_llm = clock()
        self._sent_at: dict[str, float] = {}
        self._last_order_key: str | None = None
        self._last_was_llm = False
        self._rejected_until: dict[str, float] = {}

    @staticmethod
    def _objects(value) -> list[dict]:
        return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []

    @staticmethod
    def _text_type(value) -> str:
        return str(value or "").lower()

    @staticmethod
    def _order_key(order: dict) -> str:
        if order.get('cmd') in {'attack', 'attack_move'}:
            # A changing list of unit ids should not reset the attack cooldown.
            return 'attack:group'
        return f"{order.get('cmd')}:{order.get('item', '')}:{order.get('units', '')}"

    def _can_send(self, order: dict, interval: float) -> bool:
        key = self._order_key(order)
        now = self._clock()
        return (
            now - self._sent_at.get(key, float("-inf")) >= interval
            and now >= self._rejected_until.get(key, 0)
        )

    def _heuristic(self, raw: dict) -> tuple[dict, str, float] | None:
        obs = raw.get("observation")
        if not isinstance(obs, dict):
            return None
        units = self._objects(obs.get("units"))
        buildings = self._objects(obs.get("buildings"))
        production = self._objects(obs.get("production"))
        available = {
            str(v).lower() for v in (obs.get("available_production") or [])
            if isinstance(v, str)
        }
        names = [self._text_type(b.get("type")) for b in buildings]
        queued = [self._text_type(p.get("item")) for p in production]
        legal = raw.get("legal_actions")
        legal = legal if isinstance(legal, dict) else {}

        # Deploy only a currently existing MCV. Never reuse the old unit ID.
        for unit in units:
            if self._text_type(unit.get("type")) == "mcv" and type(unit.get("id")) is int:
                order = {"cmd": "deploy", "units": [unit["id"]]}
                if self._can_send(order, 7):
                    return order, "Deploying the MCV to unlock construction.", 7

        # Placing a ready structure unblocks the build queue. The game accepts
        # omitted 'at' and chooses a valid placement near the base.
        for p in production:
            item = self._text_type(p.get("item"))
            if p.get("ready") is True and item in {"powr","tent","barr","proc","weap","afld","silo","dome","stek","atek","hpad","fix","gun","ftur","tsla","sam","agun","mslo","iron","pdox"}:
                order = {"cmd": "place", "item": item}
                if self._can_send(order, 7):
                    return order, f"Placing the completed {item}.", 7

        economy = obs.get("economy")
        economy = economy if isinstance(economy, dict) else {}
        power = economy.get("power_provided")
        drained = economy.get("power_drained")
        if (isinstance(power, (int, float)) and isinstance(drained, (int, float))
                and power < drained and "powr" in available and "powr" not in queued):
            order = {"cmd": "build", "item": "powr"}
            if self._can_send(order, 12):
                return order, "Restoring power capacity.", 12

        # Avoid duplicate queued structures and honor available production.
        # The two barracks names depend on faction.
        for options in (("powr",), ("tent", "barr"), ("proc",), ("weap",)):
            if any(t in names for t in options) or any(t in queued for t in options):
                continue
            item = next((t for t in options if t in available), None)
            if item:
                order = {"cmd": "build", "item": item}
                if self._can_send(order, 12):
                    return order, f"Building {item} to develop the base.", 12

        # Respond to an authoritative, ready-to-send attack order when there
        # is a meaningful group of combat units; don't attack with harvesters.
        combat = [u for u in units if self._text_type(u.get("type")) not in
                  {"mcv", "harv", "harvester", "truk", "truck", "miner"}]
        attack = legal.get("attack_now")
        if len(combat) >= 4 and isinstance(attack, dict) and attack.get("cmd"):
            if self._can_send(attack, 38):
                return attack, "Advancing an assembled force toward the enemy.", 38

        # Produce combat units only when there is basic economic infrastructure.
        if "proc" in names:
            priority = ("3tnk", "2tnk", "1tnk", "e1")
            for item in priority:
                if item in available:
                    order = {"cmd": "train", "item": item, "count": 2}
                    if self._can_send(order, 18):
                        return order, f"Training {item} to strengthen the army.", 18
        return None

    def choose_action(self, state, context):
        raw = state.raw
        if not isinstance(raw.get("observation"), dict):
            return WAIT
        now = self._clock()
        # Never block an immediate, deterministic order waiting for the LLM.
        heuristic = self._heuristic(raw)
        if heuristic is not None:
            order, reason, interval = heuristic
            key = self._order_key(order)
            self._sent_at[key] = self._clock()
            self._last_order_key = key
            self._last_was_llm = False
            self._log(f"[riffo] Red Alert: {reason}")
            return WithReasoning({"type": "orders", "orders": [order]}, reason)
        # The model advises only when there is no immediately actionable rule.
        # A short request timeout prevents AI latency from dominating realtime play.
        if now - self._last_llm >= self.LLM_INTERVAL and self._llm.calls < self.MAX_LLM_CALLS:
            self._last_llm = now
            action = self._llm.choose_action(state, context)
            if action is not WAIT:
                self._last_was_llm = True
                return action
        return WAIT

    def on_action_result(self, result, context):
        if self._last_was_llm:
            self._llm.on_action_result(result, context)
        elif self._last_order_key and (
            result.get("error") == "INVALID_ACTION" or
            result.get("accepted") is False
        ):
            self._rejected_until[self._last_order_key] = self._clock() + 30
        self._last_order_key = None



# Pokémon legal action templates are authoritative. Our local tactical policy
# always selects among those legal options; it never guesses a Showdown command.
# No API requests are needed during Pokémon, so it cannot hit Groq 429 / 413.
_PKM_PRIORITY = {
    "calyrexshadow": 112, "calyrexice": 106, "fluttermane": 108,
    "incineroar": 104, "urshifurapidstrike": 103, "urshifusinglestrike": 101,
    "rillaboom": 101, "tornadus": 100, "ogerponwellspring": 100,
    "amoonguss": 97, "landorustherian": 97, "chienpao": 97,
    "ironhands": 95, "dialga": 95, "kyogre": 95, "groudon": 95,
    "whimsicott": 94, "indeedeef": 93, "farigiraf": 92,
    "kingambit": 91, "gholdengo": 91, "dragonite": 90,
    "pelipper": 88, "archaludon": 91, "ursalunabloodmoon": 92,
    "garchomp": 87, "galarianmoltres": 87, "arcanine": 86,
    "regieleki": 88, "regidrago": 87, "heatran": 89,
    "porygon2": 89, "cresselia": 88, "murkrow": 85,
    "terapagos": 102, "zamazenta": 99, "miraidon": 105, "koraidon": 103,
}

_MOVE_BONUS = {
    "fakeout": 65, "tailwind": 80, "spore": 91, "sleeppowder": 65,
    "followme": 63, "ragepowder": 63, "icywind": 67,
    "electroweb": 61, "snarl": 53, "partingshot": 66,
    "knockoff": 55, "helpinghand": 45, "trickroom": 57,
    "willowisp": 53, "thunderwave": 47,
    "surgingstrikes": 93, "wickedblow": 90,
    "astralbarrage": 87, "glaciallance": 87,
    "dazzlinggleam": 53, "heatwave": 54, "rockslide": 56,
    "makeitrain": 76, "bleakwindstorm": 73,
    "closecombat": 67, "headlongrush": 71, "earthpower": 56,
    "moonblast": 61, "flareblitz": 64, "woodhammer": 59,
    "wildcharge": 55, "playrough": 56,
    "uturn": 44, "voltswitch": 44,
    "protect": 15, "detect": 15, "wideguard": 30,
    "calmmind": 31, "swordsdance": 32, "nastyplot": 32,
}

# Known species and move types improve targeting without requiring an API call.
# Unknown species or moves are treated neutrally rather than guessed.
_TYPES = {
    "incineroar": ("fire", "dark"), "tornadus": ("flying",),
    "rillaboom": ("grass",), "indeedeef": ("psychic", "normal"),
    "fluttermane": ("ghost", "fairy"), "urshifurapidstrike": ("fighting", "water"),
    "urshifusinglestrike": ("fighting", "dark"),
    "amoonguss": ("grass", "poison"), "landorustherian": ("ground", "flying"),
    "chienpao": ("dark", "ice"), "ironhands": ("electric", "fighting"),
    "farigiraf": ("normal", "psychic"), "gholdengo": ("steel", "ghost"),
    "kingambit": ("dark", "steel"), "dragonite": ("dragon", "flying"),
    "pelipper": ("water", "flying"), "archaludon": ("steel", "dragon"),
    "heatran": ("fire", "steel"), "kyogre": ("water",),
    "groudon": ("ground",), "calyrexshadow": ("psychic", "ghost"),
    "calyrexice": ("psychic", "ice"), "miraidon": ("electric", "dragon"),
    "koraidon": ("fighting", "dragon"), "zamazenta": ("fighting",),
    "ogerponwellspring": ("grass", "water"), "garchomp": ("dragon", "ground"),
    "whimsicott": ("grass", "fairy"), "ursalunabloodmoon": ("ground", "normal"),
}
_MOVE_TYPES = {
    "moonblast": "fairy", "dazzlinggleam": "fairy", "playrough": "fairy",
    "surgingstrikes": "water", "hydropump": "water", "waterfall": "water",
    "muddywater": "water", "scald": "water", "originpulse": "water",
    "heatwave": "fire", "flareblitz": "fire", "flamethrower": "fire",
    "overheat": "fire", "fireblast": "fire",
    "woodhammer": "grass", "grassyglide": "grass", "leafstorm": "grass",
    "powerwhip": "grass", "energyball": "grass",
    "closecombat": "fighting", "drainpunch": "fighting", "sacredsword": "fighting",
    "wickedblow": "dark", "knockoff": "dark", "darkpulse": "dark",
    "astralbarrage": "ghost", "shadowball": "ghost", "shadowclaw": "ghost",
    "glaciallance": "ice", "icebeam": "ice", "blizzard": "ice",
    "freezedry": "ice", "iciclecrash": "ice",
    "earthpower": "ground", "earthquake": "ground", "headlongrush": "ground",
    "stompingtantrum": "ground", "highhorsepower": "ground",
    "thunderbolt": "electric", "electroweb": "electric", "wildcharge": "electric",
    "thunderclap": "electric", "discharge": "electric",
    "makeitrain": "steel", "flashcannon": "steel", "ironhead": "steel",
    "rockslide": "rock", "stoneedge": "rock", "powergem": "rock",
    "bleakwindstorm": "flying", "hurricane": "flying", "bravebird": "flying",
    "sludgebomb": "poison", "sludgewave": "poison",
    "psychic": "psychic", "psyshock": "psychic", "expandingforce": "psychic",
    "dragonpulse": "dragon", "dracometeor": "dragon", "dragonclaw": "dragon",
    "fakeout": "normal", "extremespeed": "normal", "hypervoice": "normal",
}
_SUPER_EFFECTIVE = {
    "fire": {"grass", "ice", "bug", "steel"},
    "water": {"fire", "ground", "rock"},
    "grass": {"water", "ground", "rock"},
    "electric": {"water", "flying"},
    "ice": {"grass", "ground", "flying", "dragon"},
    "fighting": {"normal", "ice", "rock", "dark", "steel"},
    "ground": {"fire", "electric", "poison", "rock", "steel"},
    "fairy": {"fighting", "dragon", "dark"},
    "ghost": {"ghost", "psychic"},
    "dark": {"ghost", "psychic"},
    "rock": {"fire", "ice", "flying", "bug"},
    "steel": {"ice", "rock", "fairy"},
    "flying": {"grass", "fighting", "bug"},
    "poison": {"grass", "fairy"},
    "psychic": {"fighting", "poison"},
    "dragon": {"dragon"},
}
_IMMUNE = {
    "electric": {"ground"}, "ground": {"flying"},
    "normal": {"ghost"}, "fighting": {"ghost"},
    "ghost": {"normal"}, "psychic": {"dark"},
    "dragon": {"fairy"}, "poison": {"steel"},
}


def _target_matchup_bonus(option: dict, target: int) -> float:
    move_type = _name(option.get("move_type") or option.get("element") or _MOVE_TYPES.get(_name(option.get("move_id"))))
    if not move_type:
        return 0.0
    target_options = option.get("target_options")
    if not isinstance(target_options, list):
        return 0.0
    species = next((item.get("species") for item in target_options
                    if isinstance(item, dict) and item.get("target") == target), None)
    types = _TYPES.get(_name(species))
    if not types:
        return 0.0
    if any(t in _IMMUNE.get(move_type, set()) for t in types):
        return -200.0
    multiplier = sum(t in _SUPER_EFFECTIVE.get(move_type, set()) for t in types)
    return multiplier * 26.0


_SINGLE_USE_EARLY = {"tailwind", "trickroom"}
_SUPPORT_ALLY = {"helpinghand", "coaching", "pollenpuff", "decorate", "healpulse"}
_SPREAD = {"heatwave", "dazzlinggleam", "rockslide", "muddywater", "blizzard", "snarl", "icywind", "astralbarrage", "glaciallance", "makeitrain", "bleakwindstorm"}
_AVOID = {"explosion", "selfdestruct", "memento", "finalgambit", "perishsong"}


def _name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _species_score(species: Any) -> float:
    key = _name(species)
    if key in _PKM_PRIORITY:
        return _PKM_PRIORITY[key]
    # Card pools can label forms with suffixes; value remains a rough ranking.
    for prefix, score in _PKM_PRIORITY.items():
        if key.startswith(prefix):
            return score - 2
    return 60.0


class PokemonCoach:
    """Deterministic, near-instant Pokémon policy with server-validated moves."""

    def __init__(self, *, log=print):
        self.turn = 0
        self.used_setup: set[str] = set()
        self.last_moves: dict[int, str] = {}
        self.log = log

    @staticmethod
    def _template(action, kind):
        item = action.input.get("action")
        if not isinstance(item, dict) or item.get("type") != kind:
            raise ValueError("Unexpected Pokémon legal-action template")
        return item

    def _draft(self, state):
        obs = state.raw.get("observation")
        obs = obs if isinstance(obs, dict) else {}
        pool = obs.get("available_cards")
        card_by_id = {
            str(item.get("card_id")): item
            for item in pool if isinstance(item, dict)
        } if isinstance(pool, list) else {}

        def score(action):
            card_id = action.action_id.split(":", 1)[-1]
            card = card_by_id.get(card_id, {})
            pokemon = card.get("species") or card.get("name") or action.label or ""
            priority = _species_score(pokemon)
            # Avoid hard-coding hidden abilities or moves: only use observed card fields.
            moves = " ".join(map(str, card.get("moves", []))).lower()
            if "tailwind" in moves or "fake out" in moves:
                priority += 4
            return priority

        action = max(state.legal_actions, key=score)
        return action, "Drafting a versatile, high-impact Pokémon."

    def _lineup(self, template):
        roster = template.get("roster") or []
        if len(set(roster)) < 4:
            raise ValueError("Too few distinct legal Pokémon in team preview")
        bring = sorted(set(roster), key=_species_score, reverse=True)[:4]
        speed = next((s for s in bring if _name(s) in {"tornadus", "whimsicott", "murkrow", "cresselia"}), None)
        if speed:
            leads = [speed, next(s for s in bring if s != speed)]
        else:
            leads = bring[:2]
        return {"type": "select_lineup", "bring": bring, "leads": leads}, "Leading with complementary high-value Pokémon."

    def _score_move(self, option, slot, target, last_target, reserved_setup=None):
        kind = option.get("type")
        if kind == "pass":
            return -500
        if kind == "switch":
            return -35 + _species_score(option.get("species")) / 15
        if kind != "move":
            return -1000
        move = _name(option.get("move_id"))
        power = option.get("base_power", option.get("power", 0))
        power = power if isinstance(power, (int, float)) else 0
        base = 32 + min(85, max(0, power)) / 2.5
        base += _MOVE_BONUS.get(move, 0)
        if move in _AVOID:
            base -= 110
        if move == "fakeout":
            base += 26 if self.turn == 0 else -43
        if move in _SINGLE_USE_EARLY:
            base += 30 if self.turn <= 1 and move not in self.used_setup else -90
            if move == reserved_setup:
                base -= 180
        if move in {"protect", "detect"}:
            base -= 22
            if self.last_moves.get(slot) in {"protect", "detect"}:
                base -= 38
        if move in _SPREAD:
            base += 13
        if move in _SUPPORT_ALLY:
            base += 15
        if target is not None and target > 0:
            base += _target_matchup_bonus(option, target)
            base += 9
            if last_target == target and move not in _SPREAD:
                base += 11  # Coordinated fire improves knockout chances.
        if target is not None and target < 0 and move not in _SUPPORT_ALLY:
            base -= 140  # Never attack a teammate by accident.
        # Moves without listed targets (field/spread) are allowed without a target.
        return base

    def _best_slot(self, options, slot, *, forced=False, reserved_switch=None, last_target=None, reserved_setup=None):
        candidates = []
        for option in options:
            if not isinstance(option, dict):
                continue
            kind = option.get("type")
            if forced and kind != "switch":
                continue
            if kind == "switch" and option.get("species") == reserved_switch:
                continue
            if kind == "move":
                targets = option.get("targets") or []
                targets = targets if isinstance(targets, list) else []
                # Positive integers always target opponents under UCLA's doubles schema.
                if _name(option.get("move_id")) in _SUPPORT_ALLY:
                    eligible = [t for t in targets if type(t) is int and t < 0]
                    eligible = eligible or targets
                else:
                    eligible = [t for t in targets if type(t) is int and t > 0]
                    eligible = eligible or [t for t in targets if type(t) is int and t == 0]
                if targets and not eligible:
                    continue
                for target in (eligible or [None]):
                    payload = {"type": "move", "move_id": option["move_id"]}
                    if target is not None:
                        payload["target"] = target
                    candidates.append((self._score_move(option, slot, target, last_target, reserved_setup), payload))
            elif kind == "switch" and option.get("species"):
                payload = {"type": "switch", "species": option["species"]}
                score = 300 + _species_score(option["species"]) / 15 if forced else -15
                candidates.append((score, payload))
            elif kind == "pass":
                candidates.append((-999, {"type": "pass"}))
        if not candidates:
            raise ValueError("No valid slot action found")
        return max(candidates, key=lambda p: p[0])[1]

    def _doubles(self, template):
        slots = template.get("slots")
        if not isinstance(slots, list) or len(slots) != 2:
            raise ValueError("Unexpected doubles slot structure")
        by_slot = {int(s.get("slot", i)): s for i, s in enumerate(slots) if isinstance(s, dict)}
        if set(by_slot) != {0, 1}:
            raise ValueError("Doubles slots 0 and 1 missing")
        s0 = by_slot[0]
        choice0 = self._best_slot(s0.get("options") or [], 0, forced=bool(s0.get("force_switch")))
        reserved = choice0.get("species") if choice0["type"] == "switch" else None
        last_target = choice0.get("target") if choice0["type"] == "move" else None
        s1 = by_slot[1]
        choice1 = self._best_slot(s1.get("options") or [], 1, forced=bool(s1.get("force_switch")),
                                  reserved_switch=reserved, last_target=last_target,
                                  reserved_setup=_name(choice0.get("move_id")))
        if choice0["type"] == choice1["type"] == "pass":
            # The server allows two passes only when both slots have no other choices.
            raise ValueError("Unexpected double pass")
        for index, choice in enumerate((choice0, choice1)):
            if choice["type"] == "move":
                move = _name(choice["move_id"])
                self.last_moves[index] = move
                if move in _SINGLE_USE_EARLY:
                    self.used_setup.add(move)
        self.turn += 1
        return {"type": "doubles_turn", "slot_0": choice0, "slot_1": choice1}, "Prioritizing valid moves, tempo and coordinated knockouts."

    def choose_action(self, state, context):
        if not state.legal_actions:
            return smoke_agent.choose_action(state, context)
        try:
            first = state.legal_actions[0]
            if first.action_id == "select_lineup":
                action, reason = self._lineup(self._template(first, "select_lineup"))
            elif first.action_id == "doubles_turn":
                action, reason = self._doubles(self._template(first, "doubles_turn"))
            elif state.phase == "draft" or first.action_id.startswith("draft_pick:"):
                action, reason = self._draft(state)
            else:
                action, reason = smoke_agent.choose_action(state, context), "Valid fallback action."
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            # The official smoke agent fills in Pokémon templates with known-valid
            # moves; much safer than guessing a payload when the server schema shifts.
            self.log(f"[riffo] Pokémon policy fallback ({type(exc).__name__})")
            action = smoke_agent.choose_action(state, context)
            reason = "Server-validated fallback move."
        self.log(f"[riffo] Pokémon: {reason}")
        return WithReasoning(action, reason)


class WerewolfCoach:
    """Deterministic seven-seat Werewolf policy; never requires an API request.

    The server supplies legal target IDs; all moves are selected from them.
    Only the player's *private* observation can establish their role or
    an investigation result. Public claims are merely weak evidence.
    """

    def __init__(self, *, log=print):
        self.log = log
        self.checked: set[int] = set()
        self.seer_results: dict[int, str] = {}
        self.messages: list[tuple[int | None, str]] = []
        self.message_ids: set[int] = set()
        self.last_spoken: int | None = None
        self.wolf_mates: set[int] = set()
        self.name_positions: dict[str, int] = {}

    @staticmethod
    def _text(obs) -> str:
        if isinstance(obs, str):
            return obs
        if isinstance(obs, dict):
            # The entire observation is private to the current player.
            return json.dumps(obs, ensure_ascii=False, default=str)
        return ""

    @staticmethod
    def _player_num(value) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if 0 <= value <= 6 else None
        if isinstance(value, str):
            m = re.search(r"(?:player\s*)?([0-6])\b", value, flags=re.IGNORECASE)
            return int(m.group(1)) if m else None
        return None

    def _seat(self, value) -> int | None:
        number = self._player_num(value)
        if number is not None:
            return number
        return self.name_positions.get(str(value or "").strip().casefold())

    @staticmethod
    def _role_from_obs(obs) -> str | None:
        if isinstance(obs, dict):
            for key in ("your_role", "my_role", "role"):
                value = obs.get(key)
                if isinstance(value, str) and value.lower().strip() in {"wolf", "werewolf", "seer", "villager"}:
                    return "wolf" if "wolf" in value.lower() else value.lower().strip()
        text = WerewolfCoach._text(obs)
        m = re.search(r"\b(?:your role\s*(?:is|:)|you are (?:a |the )?)\s*(?:an?\s+)?(werewolf|wolf|seer|villager)\b", text, re.I)
        if m:
            role = m.group(1).lower()
            return "wolf" if "wolf" in role else role
        return None

    def _learn_private(self, obs, role, me):
        if role is None:
            return
        text = self._text(obs)
        if role == "wolf":
            # Only an explicitly private teammate statement establishes an ally.
            if isinstance(obs, dict):
                for key in ("wolf_teammate", "wolf_partner", "other_wolf", "teammate_position"):
                    ally = self._seat(obs.get(key))
                    if ally is not None and ally != me:
                        self.wolf_mates.add(ally)
            for pattern in (
                r"(?:your|my)\s+(?:wolf\s+)?(?:teammate|partner|fellow\s+wolf)\s*(?:is|:)?\s*Player\s*([0-6])",
                r"(?:the other wolf|your fellow wolf)\s*(?:is|:)?\s*Player\s*([0-6])",
                r"(?:your|the)\s+wolves\s*(?:are|:)\s*Player\s*([0-6])\s*(?:and|,)\s*Player\s*([0-6])",
            ):
                for match in re.finditer(pattern, text, re.I):
                    for num in match.groups():
                        if num is not None and int(num) != me:
                            self.wolf_mates.add(int(num))
        if role != "seer":
            return
        if isinstance(obs, dict):
            # Read only private observation fields, never public game_state.
            for key in ("seer_results", "investigations", "investigation_results"):
                data = obs.get(key)
                if isinstance(data, dict):
                    rows = [{"target": who, "role": result} for who, result in data.items()]
                elif isinstance(data, list):
                    rows = [x for x in data if isinstance(x, dict)]
                else:
                    rows = []
                for row in rows:
                    target = self._seat(row.get("target", row.get("player")))
                    result = str(row.get("role", row.get("result", ""))).lower()
                    if target is not None and result in {"wolf", "werewolf", "villager", "seer"}:
                        self.seer_results[target] = "wolf" if "wolf" in result else "villager"
                        self.checked.add(target)
        # These phrases are facts in our own private seer observation. Never
        # interpret public player messages as verified investigation results.
        patterns = (
            r"(?:you\s+)?investigat(?:ed|ion(?: result)?)\s*(?:of\s+)?Player\s*([0-6])[^.\n]{0,100}?\b(werewolf|wolf|villager|seer)\b",
            r"(?:seer\s+result|your\s+result)\s*[:\-]?\s*Player\s*([0-6])[^.\n]{0,60}?\b(werewolf|wolf|villager|seer)\b",
            r"Player\s*([0-6])\s*(?:was\s+)?(?:revealed|identified|confirmed|found)\s+(?:as\s+)?(?:a\s+)?(werewolf|wolf|villager|seer)\b",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                target, result = int(match.group(1)), match.group(2).lower()
                if target != me:
                    self.seer_results[target] = "wolf" if "wolf" in result else "villager"
                    self.checked.add(target)

    def _remember_messages(self, state):
        for msg in getattr(state, "new_messages", []) or []:
            index = getattr(msg, "index", None)
            if index is not None and index in self.message_ids:
                continue
            if index is not None:
                self.message_ids.add(index)
            content = str(getattr(msg, "content", "") or "")[:400]
            sender = self._seat(getattr(msg, "sender", None))
            if content:
                self.messages.append((sender, content))
        self.messages = self.messages[-70:]

    def _confirmed_dead(self, game_state):
        dead = {}
        for row in game_state.get("dead", []) or []:
            if isinstance(row, dict):
                who = self._seat(row.get("position", row.get("name", row.get("player"))))
                role = str(row.get("role", "")).lower()
                if who is not None:
                    dead[who] = role
        return dead

    def _public_suspicion(self, candidate, dead):
        """Weak evidence from *targeted* claims, not unrelated names in a message.

        A phrase like "Player0 is confirmed innocent. Vote Player1" must
        NEVER mark Player0 suspicious merely because "vote" is elsewhere.
        Role claims are unverified unless learned privately as the seer.
        """
        score = 0.0
        mention = re.compile(rf"\bPlayer\s*{candidate}\b", re.I)
        suspect_words = re.compile(r"\b(wolf|werewolf|suspicious|suspect|lying|liar|guilty|accuse|accusing|odd behavior|odd actions)\b", re.I)
        innocence_words = re.compile(r"\b(innocent|cleared|confirmed (?:as )?(?:a )?villager|not (?:a )?wolf|trustworthy)\b", re.I)
        vote_target = re.compile(rf"\b(?:vote|voting|lynch)\s+(?:for |on |against )?Player\s*{candidate}\b", re.I)
        avoid_vote = re.compile(rf"\b(?:do not|don't|never|avoid)\s+vote\s+(?:for |on )?Player\s*{candidate}\b", re.I)
        for sender, content in self.messages:
            if sender == candidate:
                # Do not count a player mentioning their own name as external evidence.
                continue
            weight = 0.15 if sender in dead and 'wolf' in dead[sender] else 1.0
            for clause in re.split(r'[.!?;\n]+', content):
                if not mention.search(clause):
                    continue
                if innocence_words.search(clause) or avoid_vote.search(clause):
                    score -= 0.6 * weight
                elif suspect_words.search(clause) or vote_target.search(clause):
                    score += 1.2 * weight
        return min(5.0, max(-3.0, score))

    def _vote_history_suspicion(self, target, game_state, dead):
        """Weakly weigh public votes against players later revealed dead.

        Supported vote_history forms are individual rows or day records with
        votes as a list or map. Unknown forms are ignored, never guessed.
        """
        history = game_state.get("vote_history") or []
        if not isinstance(history, list):
            return 0.0
        votes = []
        for entry in history[-10:]:
            if not isinstance(entry, dict):
                continue
            bundle = entry.get("votes")
            if isinstance(bundle, dict):
                votes.extend((who, whom) for who, whom in bundle.items())
            elif isinstance(bundle, list):
                for row in bundle:
                    if isinstance(row, dict):
                        votes.append((row.get("voter", row.get("player")),
                                      row.get("target", row.get("vote"))))
            elif "voter" in entry or "player" in entry:
                votes.append((entry.get("voter", entry.get("player")),
                              entry.get("target", entry.get("vote"))))
        score = 0.0
        for who, whom in votes[-70:]:
            if self._seat(who) != target:
                continue
            voted = self._seat(whom)
            if voted is not None and voted in dead:
                if "wolf" in dead[voted]:
                    score -= 0.8
                elif dead[voted] in {"villager", "seer"}:
                    score += 0.35
        return max(-3.0, min(3.0, score))

    @staticmethod
    def _tie_break(me, day, candidate):
        # Stable and seat-dependent, so seven self-play seats don't all choose
        # the same victim simply because the first legal action is Player0.
        return ((candidate * 29 + (me or 0) * 17 + day * 13 + 7) % 37) / 100.0

    def _context(self, state, context):
        raw = state.raw if isinstance(state.raw, dict) else {}
        gs = raw.get("game_state") if isinstance(raw.get("game_state"), dict) else {}
        for p in raw.get("players", []) or []:
            if isinstance(p, dict) and isinstance(p.get("position"), int) and isinstance(p.get("name"), str):
                self.name_positions[p["name"].strip().casefold()] = p["position"]
        me = self._seat(raw.get("your_position"))
        if me is None:
            me = self._seat(getattr(context, "seat_position", None))
        obs = raw.get("observation", getattr(state, "observation", ""))
        role = self._role_from_obs(obs)
        day_value = gs.get("day", 1)
        try:
            day = max(1, int(day_value))
        except (TypeError, ValueError):
            day = 1
        phase = str(gs.get("phase", ""))
        self._learn_private(obs, role, me)
        self._remember_messages(state)
        return raw, gs, me, role, day, phase

    def choose_action(self, state, context):
        raw, gs, me, role, day, phase = self._context(state, context)
        legal = []
        for action in state.legal_actions:
            seat = self._player_num(action.action_id)
            if seat is not None:
                legal.append((seat, action))
        if not legal:
            # Pass through a future, unexpected action schema, without guessing.
            return state.legal_actions[0] if state.legal_actions else WAIT

        dead = self._confirmed_dead(gs)
        # Never intentionally vote for ourselves or a known wolf teammate.
        # Keep server-provided legal actions intact if filtering leaves none.
        safe = [(n, a) for n, a in legal if n != me and not (role == 'wolf' and n in self.wolf_mates)]
        if safe:
            legal = safe
        non_abstain = [(n, action) for n, action in legal if n != 7]
        if phase == "day_vote" and non_abstain:
            legal = non_abstain  # The three-day no-elimination rule makes abstention risky.

        def score(target):
            value = self._tie_break(me, day, target)
            if target in dead:
                return -10000.0
            if phase == "night_seer":
                value += 5 if target not in self.checked else -100
                # Check influential players rather than wasting repeat scans.
                value += sum(sender == target for sender, _ in self.messages) * 0.45
            elif phase == "night_wolf":
                value += sum(sender == target for sender, _ in self.messages) * 0.9
                if any(sender == target and re.search(r"\b(i am|i'm|im) (the |a )?seer\b|\bmy investigation\b", msg, re.I)
                       for sender, msg in self.messages):
                    value += 9
            else:
                if target in self.wolf_mates and role == "wolf":
                    return -5000.0
                if role == "seer" and self.seer_results.get(target) == "wolf":
                    value += 1000.0
                if role == "seer" and self.seer_results.get(target) == "villager":
                    return -5000.0
                # For villagers, public accusation is weak evidence. For
                # wolves, eliminate vocal opponents instead of a known ally.
                value += self._public_suspicion(target, dead)
                value += self._vote_history_suspicion(target, gs, dead)
                if role == "wolf":
                    value += 0.25 * sum(sender == target for sender, _ in self.messages)
            return value

        target, decision = max(legal, key=lambda pair: score(pair[0]))
        if phase == "night_seer":
            self.checked.add(target)
            reason = "Investigating a player I have not checked before."
        elif phase == "night_wolf":
            reason = "Choosing a night target based on available evidence."
        else:
            reason = "Voting based on the current evidence and revealed roles."
        self.log(f"[riffo] Werewolf: {phase or 'move'} => Player{target}")
        return WithReasoning(decision, reason)

    def choose_message(self, state, context):
        raw, gs, me, role, day, phase = self._context(state, context)
        if day == self.last_spoken:
            return TERMINATE_MESSAGING
        self.last_spoken = day
        dead = self._confirmed_dead(gs)
        alive = {self._seat(v) for v in (gs.get("alive") or [])}
        alive.discard(None)
        if role == "seer":
            wolves = sorted(t for t, result in self.seer_results.items()
                           if result == "wolf" and t not in dead and t in alive)
            if wolves:
                target = wolves[0]
                self.log(f"[riffo] Werewolf: reporting confirmed investigation to discussion")
                return SendMessage(f"I am the seer. I investigated Player{target} and learned they are a wolf. Vote Player{target}.")
        if role == "wolf" and self.wolf_mates:
            living_mates = sorted(t for t in self.wolf_mates if t not in dead and t in alive)
            targets = sorted(t for t in alive if t != me and t not in self.wolf_mates)
            if living_mates and targets:
                target = max(targets, key=lambda t: self._public_suspicion(t, dead))
                self.log("[riffo] Werewolf: private teammate coordination")
                return SendMessage(f"Let's coordinate today's votes on Player{target} if possible.", [living_mates[0]])
        # Unlike a fabricated 'Player X is a wolf' claim, this helps coordinate
        # actual evidence without leaking private information or spamming chat.
        if day == 1 and role != "wolf":
            self.log("[riffo] Werewolf: requesting evidence")
            return SendMessage("Please share evidence and voting reasons. We need a coordinated vote, not random accusations.")
        return TERMINATE_MESSAGING



# ---------------------------------------------------------------------------
# Riffo v5: AI-assisted decision optimization across all three live games.
# We maximize estimated utility, rather than minimizing API calls at the
# expense of reasoning.  The utilities are comparative heuristics, NOT
# calibrated win probabilities or a learned value function.
# ---------------------------------------------------------------------------

import itertools
import math
import os
from examples.llm.base import object_schema, InvalidChoice
from examples.llm.pokemon import doubles_choice, lineup_choice
from examples.llm.redalert import validate_orders, order_fields, _limits


class ResilientProvider:
    """OpenAI-compatible strict-JSON provider with rate-limit backoff.

    One request per strategic decision. No automatic retry storms. Each
    subprocess/seat has its own budget; this is not a cross-process limiter.
    """

    def __init__(self, provider, *, clock=time.monotonic):
        self._provider = provider
        self.model = provider.model
        self.clock = clock
        self.blocked_until = 0.0

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        if self.clock() < self.blocked_until:
            raise ProviderError('API cooling down; local policy takes over')
        try:
            return self._provider.complete_structured(
                messages, schema_name, schema, timeout=timeout)
        except ProviderError as exc:
            # Never copy raw HTTP response bodies, tokens, or credentials to logs.
            message = str(exc)
            if '429' in message:
                self.blocked_until = self.clock() + 45
            elif '413' in message:
                self.blocked_until = self.clock() + 90
            raise


def _compact(obj: Any, limit: int = 2200) -> str:
    """Truncate observations and history so a whole state isn't re-sent."""
    if isinstance(obj, str):
        return obj[:limit]
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), default=str)[:limit]
    except (TypeError, ValueError):
        return ''


def _safe_float(value: Any, default: float = 0.0) -> float:
    return float(value) if type(value) in (int, float) and math.isfinite(float(value)) else default


class AIUtilitySelector:
    """Ask a language model to choose from a small set of valid candidates.

    Each candidate has a *local expected utility* estimate. The LLM can
    reason about missing factors (opponents, opportunity cost, dynamics)
    but can NEVER introduce an unlisted action. Invalid/failed API replies
    return the locally highest-utility candidate.
    """

    def __init__(self, provider, *, max_calls: int, log=print):
        self.provider = provider
        self.max_calls = max_calls
        self.calls = 0
        self.log = log

    def rank(self, game: str, candidates: list[dict], observation: dict,
             *, timeout: float = 8.0) -> tuple[dict, bool]:
        assert candidates
        local = max(candidates, key=lambda c: c['utility'])
        if len(candidates) <= 1 or self.calls >= self.max_calls:
            return local, False
        display = [
            {'index': idx, 'estimated_utility': round(c['utility'], 2),
             'plan': _compact(c['description'], 360)}
            for idx, c in enumerate(candidates)
        ]
        schema = object_schema({
            'index': {'type': 'integer', 'enum': list(range(len(candidates)))},
            'justification': {'type': 'string'},
        }, reasoning=False)
        payload = {
            'game': game,
            'objective': 'MAXIMIZE expected utility = reward - risk - opportunity cost; '
                         'consider strategic future states, not merely the largest heuristic score',
            'constraints': 'Choose EXACTLY one index supplied. Scores are approximate, not facts. '
                           'Treat opponent statements as uncertain. Never invent actions.',
            'observation': observation,
            'candidates': display,
        }
        system = (
            'You are Riffo, an expert multiplayer-game decision analyst. '
            'Use consequentialist cost-benefit comparison: probability of success, '
            'future positioning, resource cost, downside risk, and information gain. '
            'Choose one valid numbered candidate, returning strict JSON only. '
            'Do not treat unverified claims as ground truth. Do not expose private '
            'information in your brief justification.'
        )
        self.calls += 1
        try:
            result = self.provider.complete_structured(
                [{'role': 'system', 'content': system},
                 {'role': 'user', 'content': _compact(payload, 6000)}],
                'riffo_rank', schema, timeout=timeout,
            )
            index = result.get('index')
            if type(index) is not int or not (0 <= index < len(candidates)):
                raise ValueError('Model index was not among supplied alternatives')
            chosen = candidates[index]
            # Refuse a wildly inferior candidate when the local policy has
            # strong evidence; still give AI freedom among plausible moves.
            if chosen['utility'] < local['utility'] - 45:
                self.log('[riffo] AI proposed high-risk alternative; keeping local choice')
                return local, False
            self.log(f'[riffo] AI {game}: selected candidate {index} '
                     f'(local utility={chosen["utility"]:.1f})')
            return chosen, True
        except (ProviderError, ValueError, TypeError, KeyError) as exc:
            self.log(f'[riffo] AI {game}: {type(exc).__name__}; using local utility fallback')
            return local, False

    def draft_message(self, observation: dict, *, timeout: float = 8.0) -> str | None:
        if self.calls >= self.max_calls:
            return None
        self.calls += 1
        schema = object_schema({'message': {'type': 'string'},
                                'send': {'type': 'boolean'}}, reasoning=False)
        try:
            answer = self.provider.complete_structured(
                [
                    {'role': 'system', 'content': (
                        'You play seven-seat Werewolf. Write a short strategic in-game '
                        'message: solicit information, cite evidence, and coordinate '
                        'votes. Be skeptical of other players claims. Never pretend '
                        'an unverified rumor is a confirmed seer investigation. '
                        'Limit the message to 35 words. Output strict JSON.'
                    )},
                    {'role': 'user', 'content': _compact(observation, 4300)},
                ],
                'riffo_chat', schema, timeout=timeout,
            )
            if answer.get('send') is not True:
                return None
            message = answer.get('message')
            if not isinstance(message, str):
                return None
            return ' '.join(message.split()[:35])[:260] or None
        except (ProviderError, ValueError, KeyError, TypeError):
            return None


class HybridPokemon:
    """Choose among validated draft/lineup/turn actions using local costs and AI."""

    def __init__(self, provider, *, log=print):
        self.local = PokemonCoach(log=lambda *_: None)
        self.selector = AIUtilitySelector(provider, max_calls=int(os.getenv('RIFFO_POKEMON_AI_CALLS', '18')), log=log)
        self.log = log

    def _alternatives(self, state) -> list[dict]:
        if not state.legal_actions:
            return []
        first = state.legal_actions[0]
        if first.action_id.startswith('draft_pick:'):
            obs = state.raw.get('observation') if isinstance(state.raw, dict) else {}
            cards = obs.get('available_cards', []) if isinstance(obs, dict) else []
            pool = {str(row.get('card_id')): row for row in cards if isinstance(row, dict)}
            candidates = []
            for action in state.legal_actions:
                card = pool.get(action.action_id.partition(':')[2], {})
                species = card.get('species') or action.label or action.action_id
                score = _species_score(species)
                moves = _compact(card.get('moves', []), 250).lower()
                score += 4 if 'tailwind' in moves or 'fake out' in moves else 0
                candidates.append({'action': action, 'utility': score,
                                   'description': f'Draft {species}; card={_compact(card, 350)}'})
            return sorted(candidates, key=lambda x: -x['utility'])[:7]

        if first.action_id == 'select_lineup':
            template = first.input.get('action', {})
            roster = list(dict.fromkeys(template.get('roster') or []))
            if len(roster) < 4:
                return []
            validator = lineup_choice(first, state)
            cands = []
            for bring in itertools.combinations(roster, 4):
                base = sum(_species_score(species) for species in bring)
                for lead in itertools.combinations(bring, 2):
                    speed = any(_name(x) in {'tornadus','whimsicott','murkrow','cresselia'} for x in lead)
                    score = base + (9 if speed else 0)
                    payload = {'bring': list(bring), 'leads': list(lead)}
                    try:
                        action = validator.build(payload)
                    except InvalidChoice:
                        continue
                    cands.append({'action': action, 'utility': score,
                                  'description': f'Bring {bring}; lead {lead}'})
            return sorted(cands, key=lambda x: -x['utility'])[:7]

        if first.action_id == 'doubles_turn':
            template = first.input.get('action', {})
            slots = sorted(template.get('slots') or [], key=lambda x: x.get('slot', 0))
            if len(slots) != 2:
                return []
            validator = doubles_choice(first, state)
            slot_options = []
            for number, slot in enumerate(slots):
                options = []
                for idx, option in enumerate(slot.get('options') or []):
                    kind = option.get('type')
                    if bool(slot.get('force_switch')) and kind != 'switch':
                        continue
                    targets = option.get('targets') or []
                    if kind == 'move':
                        move = _name(option.get('move_id'))
                        targets = ([t for t in targets if type(t) is int and (
                            t >= 0 or move in _SUPPORT_ALLY)] if targets else [None])
                        if not targets:
                            continue
                        for target in targets:
                            score = self.local._score_move(option, number, target, None)
                            options.append(({'option': idx, 'target': target}, score,
                                            f'{option.get("move_id")} at {target}'))
                    elif kind == 'switch':
                        score = _species_score(option.get('species')) / 12 - 10
                        if slot.get('force_switch'):
                            score += 250
                        options.append(({'option': idx, 'target': None}, score,
                                        f'switch to {option.get("species")}'))
                    elif kind == 'pass':
                        options.append(({'option': idx, 'target': None}, -250, 'pass'))
                slot_options.append(sorted(options, key=lambda x: -x[1])[:5])
            candidates = []
            for a, b in itertools.product(*slot_options):
                try:
                    action = validator.build({'slot_0': a[0], 'slot_1': b[0]})
                except InvalidChoice:
                    continue
                coordinated = (action['slot_0'].get('target') == action['slot_1'].get('target')
                               and type(action['slot_0'].get('target')) is int
                               and action['slot_0']['target'] > 0
                               and action['slot_0']['type'] == action['slot_1']['type'] == 'move')
                score = a[1] + b[1] + (12 if coordinated else 0)
                candidates.append({'action': action, 'utility': score,
                                   'description': f'slot0: {a[2]}; slot1: {b[2]}'})
            return sorted(candidates, key=lambda x: -x['utility'])[:8]
        return []

    def choose_action(self, state, context):
        before_setup = set(self.local.used_setup)
        before_last = dict(self.local.last_moves)
        # Score BEFORE the local fallback updates its move history; otherwise
        # Tailwind and Fake Out would be penalized on their first real use.
        try:
            candidates = self._alternatives(state)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self.log(f'[riffo] Pokemon candidate generation fallback ({type(exc).__name__})')
            candidates = []
        fallback = self.local.choose_action(state, context)
        if not candidates:
            return fallback
        phase = state.phase
        # Draft/preview always warrant model consultation. During battle,
        # consult on every second turn; the local evaluator handles the rest.
        use_ai = phase == 'draft' or state.legal_actions[0].action_id == 'select_lineup' or self.local.turn % 2 == 1
        if use_ai:
            raw = state.raw if isinstance(state.raw, dict) else {}
            observation = {
                'phase': phase,
                'brief_state': _compact(raw.get('observation'), 1600),
                'our_last_moves': before_last,
                'setup_used': sorted(before_setup),
            }
            choice, _ = self.selector.rank('pokemon', candidates, observation,
                                           timeout=5.5 if phase == 'draft' else 9)
        else:
            choice = max(candidates, key=lambda x: x['utility'])
        picked = choice['action']
        if isinstance(picked, dict) and picked.get('type') == 'doubles_turn':
            self.local.used_setup = before_setup
            self.local.last_moves = before_last
            for i in (0, 1):
                move = picked[f'slot_{i}']
                if move.get('type') == 'move':
                    name = _name(move.get('move_id'))
                    self.local.last_moves[i] = name
                    if name in _SINGLE_USE_EARLY:
                        self.local.used_setup.add(name)
        reason = 'Utility-ranked, legal Pokémon decision.'
        self.log(f'[riffo] Pokemon: {reason} Model calls={self.selector.calls}')
        return WithReasoning(picked, reason)


class HybridWerewolf:
    """Local role/evidence inference plus AI interpretation and cost-sensitive votes."""

    def __init__(self, provider, *, log=print):
        self.local = WerewolfCoach(log=lambda *_: None)
        self.selector = AIUtilitySelector(provider, max_calls=int(os.getenv('RIFFO_WEREWOLF_AI_CALLS', '16')), log=log)
        self.log = log

    def _state_brief(self, state, context) -> tuple[dict, dict, int | None, str | None, int, str]:
        return self.local._context(state, context)

    def choose_action(self, state, context):
        previously_checked = set(self.local.checked)
        fallback = self.local.choose_action(state, context)
        raw, gs, me, role, day, phase = self._state_brief(state, context)
        if not isinstance(fallback, WithReasoning):
            return fallback
        dead = self.local._confirmed_dead(gs)
        options = []
        for action in state.legal_actions:
            t = self.local._player_num(action.action_id)
            if t is None or t in dead or t == me:
                continue
            if t == 7:
                # Abstaining is legal, but risky because three days with no
                # elimination make wolves win. Only allow if nothing else.
                continue
            if role == 'wolf' and t in self.local.wolf_mates:
                continue
            if role == 'seer' and phase == 'day_vote' and self.local.seer_results.get(t) == 'villager':
                continue
            suspicion = self.local._public_suspicion(t, dead)
            voting = self.local._vote_history_suspicion(t, gs, dead)
            voice = sum(sender == t for sender, _ in self.local.messages)
            if phase == 'night_seer':
                # Information gain: high for uninvestigated, influential players.
                cost = 25 if t in previously_checked else 0
                utility = 13 + min(voice, 6) - cost
            elif phase == 'night_wolf':
                seer_claim = any(sender == t and re.search(
                    r'\b(i am|i.m|im) (the |a )?seer\b|\bmy investigation\b', msg, re.I)
                    for sender, msg in self.local.messages)
                utility = 15 + 13 * int(seer_claim) + 1.3 * voice
            elif role == 'wolf':
                utility = 8 + 1.2 * suspicion + 0.4 * voice
            else:
                if role == 'seer' and self.local.seer_results.get(t) == 'wolf':
                    utility = 100
                else:
                    # Proxy for expected benefit: P(wolf)*value of elimination
                    # minus P(villager)*opportunity cost and coalition risk.
                    prob_wolf = max(0.06, min(0.9, 0.28 + 0.085 * (suspicion + voting)))
                    utility = 15 * prob_wolf - 9 * (1 - prob_wolf)
                    utility += 0.25 * min(voice, 5)
            options.append({'action': action, 'utility': utility,
                            'description': f'Player{t}: suspicion={suspicion:.1f}, public votes={voting:.1f}, messages={voice}'})
        if not options:
            return fallback
        # A seer who has confirmed a living wolf never lets LLM override
        # that fact with an unverified accusation.
        if role == 'seer' and phase == 'day_vote':
            certain = [c for c in options if self.local.seer_results.get(int(c['action'].action_id)) == 'wolf']
            if certain:
                options = certain
        options.sort(key=lambda c: -c['utility'])
        options = options[:6]
        observation = {
            'role_private_to_you': role,
            'your_seat': me,
            'day': day,
            'phase': phase,
            'private_observation': _compact(raw.get('observation'), 1100),
            'dead_and_votes': _compact({'dead': gs.get('dead'), 'vote_history': gs.get('vote_history')}, 1400),
            'recent_public_messages': [(f'Player{who}', text[:180]) for who, text in self.local.messages[-10:]],
            'confirmed_seer_results_private': self.local.seer_results if role == 'seer' else {},
            'known_wolf_teammates_private': sorted(self.local.wolf_mates) if role == 'wolf' else [],
        }
        selected, _ = self.selector.rank('werewolf', options, observation, timeout=7)
        chosen = selected['action']
        if phase == 'night_seer':
            baseline_target = self.local._player_num(fallback.action.action_id)
            if baseline_target not in previously_checked:
                self.local.checked.discard(baseline_target)
            self.local.checked.add(int(chosen.action_id))
        summary = 'Weighed role evidence, vote likelihood and strategic downside.'
        self.log(f'[riffo] Werewolf {phase}: Player{chosen.action_id}')
        return WithReasoning(chosen, summary)

    def choose_message(self, state, context):
        """Use AI to select *fact-safe* discussion messages.

        Unrestricted LLM prose previously invented witnessed actions, accused
        the seer of being a wolf, and asked people to vote for the speaker.
        Candidate selection still uses GPT-4o-mini; candidate wording is
        validated locally, so fabricated clues cannot enter the public record.
        """
        baseline = self.local.choose_message(state, context)
        raw, gs, me, role, day, phase = self._state_brief(state, context)
        # Only private confirmed Seer results justify hard claims. Wolf chat
        # stays private to the intended teammate.
        if role == 'seer' and isinstance(baseline, SendMessage):
            return baseline
        if role == 'wolf' and isinstance(baseline, SendMessage):
            return baseline
        if self.local.last_spoken != day:
            return TERMINATE_MESSAGING
        if not hasattr(self, '_ai_spoken_days'):
            self._ai_spoken_days = set()
        if day in self._ai_spoken_days:
            return TERMINATE_MESSAGING
        self._ai_spoken_days.add(day)

        dead = self.local._confirmed_dead(gs)
        alive = {self.local._seat(v) for v in (gs.get('alive') or [])}
        alive.discard(None)
        eligible = sorted(t for t in alive if t != me and t not in dead and
                          (role != 'wolf' or t not in self.local.wolf_mates) and
                          (role != 'seer' or self.local.seer_results.get(t) != 'villager'))
        options = [
            {'action': 'Let us compare actual public votes and statements. Which specific contradiction supports a wolf accusation?',
             'utility': 11.0, 'description': 'Ask for verifiable public contradictions instead of invented evidence.'},
            {'action': 'We should coordinate one evidence-based vote instead of splitting votes. Which player has the strongest public case against them?',
             'utility': 10.5, 'description': 'Organize coalition voting without accusing a particular player.'},
            {'action': 'Please separate confirmed Seer results from guesses. Can anyone cite a concrete public vote or contradictory statement?',
             'utility': 10.0, 'description': 'Discourage fabricated role claims and request checkable facts.'},
        ]
        if eligible:
            # A question is safe, but a claim that we watched a private action
            # is not. Also never request a vote for our own seat.
            target = max(eligible, key=lambda t: (self.local._public_suspicion(t, dead),
                                                  sum(sender == t for sender, _ in self.local.messages)))
            options.append({
                'action': f'Player{target}, can you explain your public voting reasoning? I want to compare evidence before choosing a target.',
                'utility': 9.8 + self.local._public_suspicion(target, dead) * 0.4,
                'description': f'Ask Player{target} to explain their public voting behavior. No factual accusation.'
            })
        if role == 'seer' and self.local.seer_results:
            options.append({
                'action': 'Please do not treat unsupported role claims as proof. We should avoid eliminating players without evidence.',
                'utility': 10.8, 'description': 'Protect verified information without leaking private checks.'
            })
        brief = {
            'day': day, 'your_role_private': role, 'your_seat': me,
            'confirmed_seer_results_private': self.local.seer_results if role == 'seer' else {},
            'public_vote_history': _compact(gs.get('vote_history'), 900),
            'public_recent_messages': [(f'Player{who}' if who is not None else 'unknown', text[:160])
                                       for who, text in self.local.messages[-10:]],
            'purpose': 'Choose the best strategically useful and truthful DISCUSSION QUESTION. '
                       'Candidates never assert unverified game facts or ask anyone to vote for you.'
        }
        picked, used_ai = self.selector.rank('werewolf_discussion', options, brief, timeout=7)
        if used_ai:
            self.log(f'[riffo] Werewolf: AI-selected safe discussion on day {day}')
        return SendMessage(picked['action'])


class HybridRedAlert(FastRedAlert):
    """Immediate safe micro + infrequent AI military/economic plan comparison."""

    MAX_LLM_CALLS = 10
    LLM_INTERVAL = 75.0

    @staticmethod
    def _order_utility(order: dict, raw: dict) -> float:
        """Rough net benefit: progress + force + economy - risk - spending."""
        if not isinstance(order, dict):
            return -1e6
        obs = raw.get('observation') if isinstance(raw.get('observation'), dict) else {}
        buildings = [str(b.get('type')).lower() for b in obs.get('buildings', []) if isinstance(b, dict)]
        units = [u for u in obs.get('units', []) if isinstance(u, dict)]
        econ = obs.get('economy') if isinstance(obs.get('economy'), dict) else {}
        power_short = _safe_float(econ.get('power_drained')) > _safe_float(econ.get('power_provided'))
        cash = _safe_float(econ.get('cash'), 2500)
        cmd, item = order.get('cmd'), str(order.get('item') or '').lower()
        if cmd == 'deploy':
            return 1000
        if cmd == 'place':
            return 850
        if cmd == 'build':
            base = {'powr': 105, 'tent': 90, 'barr': 90, 'proc': 118,
                    'weap': 112, 'afld': 105}.get(item, 30)
            if item == 'powr' and power_short:
                base += 90
            # A larger cash buffer gives more opportunity to invest.
            base += min(20, max(-25, (cash - 1500) / 160))
            if item in buildings:
                base -= 80
            return base
        if cmd == 'train':
            combat = sum(str(u.get('type')).lower() not in {'mcv','harv','harvester'} for u in units)
            return 80 + (30 if combat < 4 else -8) - 0.04 * max(0, 1800 - cash)
        if cmd in {'attack_move', 'attack'}:
            army_size = len(order.get('units') or [])
            return 45 + min(85, army_size * 11) - (40 if army_size < 3 else 0)
        if cmd == 'repair':
            return 45
        return 5

    def choose_action(self, state, context):
        raw = state.raw
        if not isinstance(raw.get('observation'), dict):
            return WAIT
        candidate = self._heuristic(raw)
        if candidate and candidate[0].get('cmd') in {'deploy','place'}:
            return super().choose_action(state, context)
        now = self._clock()
        should_ask = (now - self._last_llm >= self.LLM_INTERVAL
                      and self._llm.calls < self.MAX_LLM_CALLS)
        if should_ask:
            self._last_llm = now
            try:
                model_action = self._llm.choose_action(state, context)
                if isinstance(model_action, WithReasoning):
                    payload = model_action.action
                    orders = payload.get('orders') if isinstance(payload, dict) else None
                    if isinstance(orders, list):
                        valid, dropped = validate_orders(orders, order_fields(context.game_config), _limits(raw))
                        if valid and not dropped:
                            model_score = sum(self._order_utility(order, raw) for order in valid)
                            local_score = self._order_utility(candidate[0], raw) if candidate else 0
                            # Prefer a viable AI multi-order plan when it adds value;
                            # otherwise keep a near-instant deterministic response.
                            if model_score >= local_score - 10:
                                self._last_was_llm = True
                                self.log_ai(model_score, local_score)
                                return model_action
            except (ProviderError, ValueError, TypeError, KeyError) as exc:
                self._log(f'[riffo] Red Alert: planning fallback ({type(exc).__name__})')
            # In case of WAIT or inferior AI choice, use the local order.
            self._last_was_llm = False
        return super().choose_action(state, context)

    def log_ai(self, score, local):
        self._log(f'[riffo] Red Alert: accepted AI plan (utility {score:.1f} vs {local:.1f})')


# ---------------------------------------------------------------------------
# Riffo v5.1 — matchup-aware scoring. Uses only authoritative offered moves.
# ---------------------------------------------------------------------------

# Gen 9 type chart: attacker -> defender -> multiplier (unspecified = 1).
_TYPE_CHART = {
  'normal': {'rock':.5,'steel':.5,'ghost':0},
  'fire': {'grass':2,'ice':2,'bug':2,'steel':2,'fire':.5,'water':.5,'rock':.5,'dragon':.5},
  'water': {'fire':2,'ground':2,'rock':2,'water':.5,'grass':.5,'dragon':.5},
  'electric': {'water':2,'flying':2,'electric':.5,'grass':.5,'dragon':.5,'ground':0},
  'grass': {'water':2,'ground':2,'rock':2,'fire':.5,'grass':.5,'poison':.5,'flying':.5,'bug':.5,'dragon':.5,'steel':.5},
  'ice': {'grass':2,'ground':2,'flying':2,'dragon':2,'fire':.5,'water':.5,'ice':.5,'steel':.5},
  'fighting': {'normal':2,'ice':2,'rock':2,'dark':2,'steel':2,'poison':.5,'flying':.5,'psychic':.5,'bug':.5,'fairy':.5,'ghost':0},
  'poison': {'grass':2,'fairy':2,'poison':.5,'ground':.5,'rock':.5,'ghost':.5,'steel':0},
  'ground': {'fire':2,'electric':2,'poison':2,'rock':2,'steel':2,'grass':.5,'bug':.5,'flying':0},
  'flying': {'grass':2,'fighting':2,'bug':2,'electric':.5,'rock':.5,'steel':.5},
  'psychic': {'fighting':2,'poison':2,'psychic':.5,'steel':.5,'dark':0},
  'bug': {'grass':2,'psychic':2,'dark':2,'fire':.5,'fighting':.5,'poison':.5,'flying':.5,'ghost':.5,'steel':.5,'fairy':.5},
  'rock': {'fire':2,'ice':2,'flying':2,'bug':2,'fighting':.5,'ground':.5,'steel':.5},
  'ghost': {'psychic':2,'ghost':2,'dark':.5,'normal':0},
  'dragon': {'dragon':2,'steel':.5,'fairy':0},
  'dark': {'psychic':2,'ghost':2,'fighting':.5,'dark':.5,'fairy':.5},
  'steel': {'ice':2,'rock':2,'fairy':2,'fire':.5,'water':.5,'electric':.5,'steel':.5},
  'fairy': {'fighting':2,'dragon':2,'dark':2,'fire':.5,'poison':.5,'steel':.5},
}
_MOVE_POWER_ESTIMATE = {
  'astralbarrage':120,'glaciallance':120,'surgingstrikes':75,'wickedblow':75,
  'makeitrain':120,'bleakwindstorm':100,'moonblast':95,'dazzlinggleam':80,
  'heatwave':95,'rockslide':75,'earthpower':90,'closecombat':120,
  'flareblitz':120,'woodhammer':120,'wildcharge':90,'playrough':90,
  'thunderbolt':90,'icebeam':90,'shadowball':80,'energyball':90,
  'knockoff':65,'grassyglide':55,'tackle':40,'fakeout':40,
  'hydropump':110,'dracometeor':130,'hypervoice':90,'sludgebomb':90,
  'earthquake':100,'stompingtantrum':75,'highhorsepower':95,
  'iciclecrash':85,'ironhead':80,'darkpulse':80,'airslash':75,
  'bravebird':120,'hurricane':110,'leafstorm':130,
  'protect':0,'detect':0,'tailwind':0,'trickroom':0,
  'spore':0,'sleeppowder':0,'followme':0,'ragepowder':0,
  'helpinghand':0,'partingshot':0,'wideguard':0,
}
_MORE_SPECIES_TYPES = {
  'arcanine':('fire',), 'porygon2':('normal',), 'cresselia':('psychic',),
  'dialga':('steel','dragon'),'terapagos':('normal',),'regieleki':('electric',),
  'regidrago':('dragon',),'chienpao':('dark','ice'), 'murkrow':('dark','flying'),
  'kingambit':('dark','steel'),'ursaring':('normal',), 'ogerponhearthflame':('grass','fire'),
  'ogerponcornerstone':('grass','rock'), 'ogerponthearthflame':('grass','fire'),
  'gholdengo':('steel','ghost'), 'ursalunabloodmoon':('ground','normal'),
  'archaludon':('steel','dragon'),'indeedee':('psychic','normal'),
  'lando':('ground','flying'),'ironbundle':('ice','water'),
  'roaringmoon':('dragon','dark'),'walkingwake':('water','dragon'),
  'chi-yu':('dark','fire'),'chiyu':('dark','fire'),
}

def _type_multiplier(move_type, target_species):
    if not move_type:
        return 1.0
    target = _name(target_species)
    defending = _TYPES.get(target) or _MORE_SPECIES_TYPES.get(target)
    if not defending:
        return 1.0
    impact = 1.0
    for defending_type in defending:
        impact *= _TYPE_CHART.get(move_type, {}).get(defending_type, 1.0)
    return impact


class MatchupPokemonCoach(PokemonCoach):
    """Net benefit = pressure, KO potential, setup tempo, safety and type matching.

    These are estimates rather than a full Showdown damage simulator. It never
    creates a new action: only scores options already offered by the server.
    """
    def __init__(self, *, log=print):
        super().__init__(log=log)
        self.active_species = {}
        self.current_target_species = {}
        self.current_target_hps = {}

    def _score_move(self, option, slot, target, last_target, reserved_setup=None):
        base = super()._score_move(option, slot, target, last_target, reserved_setup)
        if option.get('type') != 'move':
            return base
        move = _name(option.get('move_id'))
        move_type = _name(option.get('move_type') or option.get('element') or _MOVE_TYPES.get(move))
        # Subtract the old, incomplete type bonus (it incorrectly treats resisted
        # dual-type matchups as pure super-effective damage).
        if type(target) is int and target > 0:
            base -= _target_matchup_bonus(option, target)
        # Do not give a large *fixed* bonus to a move that is immune/resisted.
        base -= 0.75 * _MOVE_BONUS.get(move, 0)
        power = option.get('base_power', option.get('power'))
        if type(power) not in (int, float):
            power = _MOVE_POWER_ESTIMATE.get(move)
        if power is None:
            # A new attack move whose power is unknown should not be scored as
            # zero; novel status moves get some benefit from the existing policy.
            power = 60 if move not in _MOVE_BONUS else 0
        if power > 0:
            attacker = _name(self.active_species.get(slot, ''))
            stab = 1.5 if move_type and move_type in (_TYPES.get(attacker) or _MORE_SPECIES_TYPES.get(attacker) or ()) else 1.0
            multiplier = 1.0
            if type(target) is int and target > 0:
                species = next((x.get('species') for x in option.get('target_options', [])
                                if isinstance(x, dict) and x.get('target') == target), None)
                species = species or self.current_target_species.get(target)
                multiplier = _type_multiplier(move_type, species)
            accuracy = option.get('accuracy', 100)
            accuracy = accuracy / 100 if type(accuracy) in (int, float) and 0 < accuracy <= 100 else 1.0
            damage_proxy = float(power) * stab * multiplier * accuracy
            base += min(170, damage_proxy * 0.72)
            if multiplier == 0:
                base -= 180  # Known type immunity: avoid wasting the move.
            elif multiplier < 1:
                base -= 25
            elif multiplier >= 2:
                base += 14
            if type(target) is int and target > 0:
                hp = self.current_target_hps.get(target)
                if type(hp) in (int,float) and 0 < hp <= 1:
                    base += 22 * (1-hp) * min(2, multiplier)
        # Avoid blindly spamming early-game utility with no benefit after setup.
        if move in {'tailwind', 'trickroom'} and move in self.used_setup:
            base -= 75
        if move in {'protect', 'detect'} and self.last_moves.get(slot) in {'protect','detect'}:
            base -= 50
        if move in {'explosion', 'selfdestruct', 'memento', 'finalgambit'}:
            base -= 100
        return base


class MatchupHybridPokemon(HybridPokemon):
    """AI chooses from matchup-ranked legal alternatives, with local fallback."""
    def __init__(self, provider, *, log=print):
        super().__init__(provider, log=log)
        self.local = MatchupPokemonCoach(log=lambda *_: None)
        self.drafted_species = []

    def _alternatives(self, state):
        first = state.legal_actions[0] if state.legal_actions else None
        if first and first.action_id == 'doubles_turn':
            template = first.input.get('action') or {}
            slots = template.get('slots') or []
            self.local.active_species = {}
            self.local.current_target_species = {}
            self.local.current_target_hps = {}
            for idx, slot in enumerate(slots):
                if not isinstance(slot, dict):
                    continue
                number = slot.get('slot', idx)
                active = slot.get('active') or {}
                if isinstance(active, dict):
                    self.local.active_species[number] = active.get('species', '')
                for option in slot.get('options') or []:
                    for target in option.get('target_options') or []:
                        if isinstance(target, dict) and type(target.get('target')) is int and target['target'] > 0:
                            self.local.current_target_species[target['target']] = target.get('species', '')
            candidates = super()._alternatives(state)
            for candidate in candidates:
                action = candidate['action']
                for i in (0, 1):
                    slot_choice = action.get(f'slot_{i}', {})
                    if slot_choice.get('type') == 'move':
                        move = _name(slot_choice.get('move_id'))
                        target = slot_choice.get('target')
                        opponent = self.local.current_target_species.get(target, '')
                        kind = _name(_MOVE_TYPES.get(move))
                        candidate['description'] += (
                            f'; slot{i}: {move} on {opponent or target}'
                            f' (estimated type effectiveness {_type_multiplier(kind, opponent):g}x)')
            return candidates
        candidates = super()._alternatives(state)
        if first and first.action_id.startswith('draft_pick:'):
            # A team with 3 speed setters is not automatically stronger than
            # a balanced team. Penalize role redundancy once a role is filled.
            have_speed = any(_name(x) in {'tornadus','whimsicott','murkrow'} for x in self.drafted_species)
            have_fakeout = any(_name(x) in {'incineroar','rillaboom','ironhands'} for x in self.drafted_species)
            for candidate in candidates:
                action = candidate['action']
                obs = state.raw.get('observation') or {}
                cards = obs.get('available_cards') or []
                card = next((x for x in cards if isinstance(x, dict) and str(x.get('card_id')) == action.action_id.partition(':')[2]), {})
                species = card.get('species') or action.label or ''
                key = _name(species)
                if key in {'tornadus','whimsicott','murkrow'} and not have_speed:
                    candidate['utility'] += 9
                if key in {'tornadus','whimsicott','murkrow'} and have_speed:
                    candidate['utility'] -= 12
                if key in {'incineroar','rillaboom','ironhands'} and not have_fakeout:
                    candidate['utility'] += 6
                candidate['description'] += f'; existing team={self.drafted_species}'
            return sorted(candidates, key=lambda row: -row['utility'])
        return candidates

    def choose_action(self, state, context):
        answer = super().choose_action(state, context)
        if state.legal_actions and state.legal_actions[0].action_id.startswith('draft_pick:'):
            chosen = answer.action if isinstance(answer, WithReasoning) else answer
            label = chosen.action_id if hasattr(chosen, 'action_id') else str(chosen)
            obs = state.raw.get('observation') if isinstance(state.raw, dict) else {}
            cards = obs.get('available_cards') or [] if isinstance(obs, dict) else []
            card = next((x for x in cards if isinstance(x, dict) and str(x.get('card_id')) == label.partition(':')[2]), {})
            species = card.get('species') or getattr(chosen, 'label', None)
            if species and species not in self.drafted_species:
                self.drafted_species.append(species)
        action = answer.action if isinstance(answer, WithReasoning) else answer
        if isinstance(action, dict) and action.get('type') == 'doubles_turn':
            moves = []
            for slot_num in (0, 1):
                chosen = action.get(f'slot_{slot_num}', {})
                moves.append(f"slot{slot_num}={chosen.get('move_id', chosen.get('species', chosen.get('type')))} target={chosen.get('target', '-')}")
            self.log('[riffo] Pokemon selected: ' + ', '.join(moves))
        elif hasattr(action, 'action_id') and str(action.action_id).startswith('draft_pick:'):
            self.log('[riffo] Pokemon draft picked ' + str(getattr(action, 'label', None) or action.action_id))
        return answer




# ---------------------------------------------------------------------------
# Riffo v5.2: economy-first real-time planning and opponent-aware VGC draft.
# ---------------------------------------------------------------------------

class EconomyFirstRedAlert(HybridRedAlert):
    """Fast multi-order economic growth before a coordinated assault.

    Every decision is state-driven. A small local utility model prices the
    opportunity cost of construction/training, while GPT-4o mini periodically
    evaluates higher-level batches. No guessed unit ids or targets are sent.
    """
    MAX_LLM_CALLS = 9
    LLM_INTERVAL = 55.0
    MIN_ATTACK_FORCE = 13
    # Spend less up front to keep the refinery / factory construction path open.
    COSTS = {'powr': 300, 'tent': 300, 'barr': 300, 'proc': 2000,
             'weap': 2000, 'e1': 100, 'e3': 300, '1tnk': 700,
             '2tnk': 800, '3tnk': 950, 'harv': 1400}
    STRUCTURES = {'fact','powr','apwr','tent','barr','proc','weap',
                  'afld','silo','dome','stek','atek','hpad','fix',
                  'gun','ftur','tsla','sam','agun','mslo','iron','pdox'}
    NONCOMBAT = {'mcv','harv','harvester','truk','truck','miner','dog'}
    # Don't clog building queues with the same speculative structure.

    def __init__(self, provider, *, log=print, clock=time.monotonic):
        super().__init__(provider, log=log, clock=clock)
        self._previous_orders: list[dict] = []
        self._reported_signature = ''
        self._model_provider = provider

    @staticmethod
    def _order_key(order):
        # Defensive fire at a particular intruder must not share the long
        # cooldown for cross-map attack_move commands.
        if order.get('cmd')=='attack':
            return 'defend:'+str(order.get('target'))
        return FastRedAlert._order_key(order)

    @staticmethod
    def _valid_ids(units):
        return [u['id'] for u in units if type(u.get('id')) is int and not u.get('wreck')]

    def _planning_facts(self, raw):
        obs=raw.get('observation') if isinstance(raw.get('observation'),dict) else {}
        units=self._objects(obs.get('units'))
        buildings=self._objects(obs.get('buildings'))
        queues=self._objects(obs.get('production'))
        names=[self._text_type(b.get('type')) for b in buildings]
        queued=[self._text_type(p.get('item')) for p in queues]
        available={str(s).lower() for s in obs.get('available_production', []) if isinstance(s,str)}
        econ=obs.get('economy') if isinstance(obs.get('economy'),dict) else {}
        cash=_safe_float(econ.get('cash'), 2500)
        power=_safe_float(econ.get('power_provided'))
        drain=_safe_float(econ.get('power_drained'))
        combat=[u for u in units if self._text_type(u.get('type')) not in self.NONCOMBAT and not u.get('wreck')]
        harvesters=[u for u in units if self._text_type(u.get('type')) in {'harv','harvester','miner'} and not u.get('wreck')]
        return obs,units,names,queues,queued,available,cash,power,drain,combat,harvesters

    def _ready_to_attack(self, raw, facts=None):
        f=facts or self._planning_facts(raw)
        _,_,names,_,_,_,cash,_,_,combat,_=f
        # A few scattered rifle infantry should not be committed to a base assault.
        tank_count=sum(self._text_type(u.get('type')) in {'1tnk','2tnk','3tnk','4tnk','arty'} for u in combat)
        army_value=len(combat)+2*tank_count
        if 'proc' not in names:
            return False
        if army_value < self.MIN_ATTACK_FORCE:
            return False
        obs=f[0]
        visible=[x for x in self._objects(obs.get('visible_enemies'))
                 if self._text_type(x.get('type')) not in self.NONCOMBAT]
        enemy_armor=sum(self._text_type(x.get('type')) in {'1tnk','2tnk','3tnk','4tnk','arty'}
                        for x in visible)
        visible_enemy_value=len(visible)+2*enemy_armor
        # Attacking into a larger army is especially bad when a resource lead
        # can be preserved and converted into a late-game tiebreak victory.
        if visible_enemy_value>0 and army_value < visible_enemy_value*1.1:
            return False
        # Prefer to establish a productive base before destroying the army.
        return 'weap' in names or army_value>=22

    def _can_afford(self, item, available, cash, reserve=0):
        return item in available and cash >= self.COSTS.get(item, 500)+reserve

    def _build_plan(self, raw):
        f=self._planning_facts(raw)
        obs,units,names,queues,queued,available,cash,power,drain,combat,harvesters=f
        legal=raw.get('legal_actions') if isinstance(raw.get('legal_actions'),dict) else {}
        orders=[]; notes=[]
        now=self._clock()

        def add(order, cooldown, reason):
            if not self._can_send(order,cooldown):
                return False
            orders.append(order); notes.append(reason)
            return True

        # Only use observed ids. An MCV changes identity when deployed.
        for u in units:
            if self._text_type(u.get('type'))=='mcv' and type(u.get('id')) is int:
                add({'cmd':'deploy','units':[u['id']]}, 8, 'deploy MCV')
                break
        if orders:
            return orders,notes

        # Structures ready in construction queue must be placed to unlock the next one.
        for p in queues:
            item=self._text_type(p.get('item'))
            if p.get('ready') is True and item in self.STRUCTURES:
                if add({'cmd':'place','item':item},10, 'place '+item):
                    return orders,notes

        # Build at most one structure each observation. Cash is a scarce resource.
        has=lambda x:x in names or x in queued
        to_build=None; why=''
        if 'powr' in available and (not has('powr') or (power>0 and drain>=power*0.92 and not any(p.get('item')=='powr' for p in queues))):
            to_build='powr'; why='power capacity'
        elif not (has('tent') or has('barr')):
            to_build=next((x for x in ('tent','barr') if x in available),None);why='early infantry access'
        elif not has('proc'):
            to_build='proc';why='first refinery and income'
        elif not has('weap'):
            to_build='weap';why='vehicle production'
        elif names.count('proc')<2 and queued.count('proc')==0 and len(harvesters)<2 and cash>=2800:
            to_build='proc';why='second refinery and income'
        elif names.count('powr')<2 and queued.count('powr')==0 and power>0 and drain>=power*0.8:
            to_build='powr';why='power headroom'

        if to_build and to_build in available and to_build not in queued and self._can_afford(to_build,available,cash,0):
            add({'cmd':'build','item':to_build}, 25, 'build '+to_build+' ('+why+')')

        # Keep production going *alongside* building. Avoid e1 x2 endlessly.
        # When upgrading economy, do not burn the money reserved for proc/weap.
        reserv=0
        if not has('proc'):
            reserv=2000
        elif not has('weap'):
            reserv=1800
        if orders and orders[0]['cmd']=='build':
            new_structure = orders[0]['item']
            cash -= self.COSTS.get(new_structure, 500)
            # That expensive structure has already been funded in this batch;
            # reserving its entire price a second time would stall production.
            if new_structure in {'proc', 'weap'}:
                reserv = 500
        budget=max(0,cash-reserv)
        queue_types=[self._text_type(p.get('queue')) for p in queues]
        already_queued=sum(1 for p in queues if self._text_type(p.get('item')) in {'e1','e2','e3','1tnk','2tnk','3tnk','harv'})
        # Harvesters are critical for scaling, but choose only actually available items.
        need_harv=len(harvesters)<2 and 'weap' in names
        if need_harv and 'harv' in available and 'harv' not in queued and budget>=self.COSTS['harv']:
            add({'cmd':'train','item':'harv','count':1}, 38, 'train income harvester')
        elif already_queued < 4 and budget>=300:
            # Favor tanks once the factory is online; use rifle infantry for mass.
            if 'weap' in names:
                unit=next((x for x in ('3tnk','2tnk','1tnk') if x in available),None)
            else:
                unit=None
            if not unit:
                unit=next((x for x in ('e3','e1','e2') if x in available),None)
            if unit:
                unit_cost=self.COSTS.get(unit,150)
                count=min(10,int(budget//unit_cost))
                if count>=1:
                    # Send a substantial queue (not e1 x2 per 20 seconds).
                    add({'cmd':'train','item':unit,'count':count}, 28,
                        'train '+unit+' x'+str(count))

        # A known idle harvester should harvest, not join an assault.
        for h in harvesters[:2]:
            if h.get('idle') is True and type(h.get('id')) is int:
                add({'cmd':'harvest','units':[h['id']]},28,'resume harvesting')
                break

        # Respond to nearby enemy raids without confusing defense with a
        # premature cross-map assault. Attack a visible enemy near our base;
        # only use authoritative enemy and our-own unit IDs.
        base=obs.get('base_center')
        if isinstance(base,dict):
            base=(base.get('x'),base.get('y'))
        if isinstance(base,(tuple,list)) and len(base)>=2 and all(type(v) in (int,float) for v in base[:2]):
            foes=[]
            for enemy in self._objects(obs.get('visible_enemies')):
                pos=(enemy.get('x'),enemy.get('y'))
                if not all(type(v) in (int,float) for v in pos):
                    cell=enemy.get('cell')
                    if isinstance(cell,(tuple,list)) and len(cell)>=2:
                        pos=cell[:2]
                if type(enemy.get('id')) is int and all(type(v) in (int,float) for v in pos):
                    dist2=(pos[0]-base[0])**2+(pos[1]-base[1])**2
                    if dist2<=16**2: foes.append((dist2,enemy))
            if foes and len(combat)>=2:
                threat=min(foes,key=lambda row:row[0])[1]
                defenders=self._valid_ids(combat)[:18]
                if defenders:
                    add({'cmd':'attack','units':defenders,'target':threat['id']},20,
                        'defend base against visible enemy')

        # Hold army until it is big enough; do not march 1-4 riflemen to death.
        attack=legal.get('attack_now')
        if self._ready_to_attack(raw,f) and isinstance(attack,dict) and attack.get('cmd'):
            # Attack group is provided by the server. Never fabricate coordinates.
            if len(attack.get('units') or [])>=6:
                add(attack,48, 'coordinated attack with '+str(len(attack['units']))+' units')

        return orders[:20],notes

    def _accept_model(self, raw, model_action, local):
        if not isinstance(model_action,WithReasoning) or not isinstance(model_action.action,dict):
            return None
        candidate=model_action.action.get('orders')
        if not isinstance(candidate,list) or not candidate:
            return None
        # Check SDK syntax; then filter economically self-defeating orders.
        valid,dropped=validate_orders(candidate,order_fields(None),_limits(raw))
        if dropped or not valid:
            return None
        f=self._planning_facts(raw)
        _,units,names,queues,queued,available,cash,_,_,_,_=f
        reserved=0
        kept=[]
        for order in valid:
            cmd=order['cmd']; item=str(order.get('item') or '').lower()
            if cmd in {'attack','attack_move'} and not self._ready_to_attack(raw,f):
                continue
            if cmd=='train':
                if item not in available or not self._can_send(order,22): continue
                reserved+=self.COSTS.get(item,400)*order.get('count',1)
            elif cmd=='build':
                if item not in available or item in queued or not self._can_send(order,20): continue
                reserved+=self.COSTS.get(item,1200)
            if reserved>cash:
                continue
            kept.append(order)
        if not kept:
            return None
        # Maintain local economy steps even if AI suggests an attack-only plan.
        critical={x['cmd'] for x in local if x['cmd'] in {'deploy','place','build'}}
        if critical and not any(o['cmd'] in critical for o in kept):
            return None
        return kept

    def choose_action(self,state,context):
        raw=state.raw
        if not isinstance(raw.get('observation'),dict):
            return WAIT
        plan,notes=self._build_plan(raw)
        now=self._clock()
        # AI evaluates possible complex tactics occasionally, but local fast
        # production continues even if the model times out or gives bad orders.
        f=self._planning_facts(raw)
        established='proc' in f[2] and 'weap' in f[2]
        if (established and not any(o['cmd']=='attack' for o in plan)
                and now-self._last_llm >= self.LLM_INTERVAL and self._llm.calls < self.MAX_LLM_CALLS):
            self._last_llm=now
            try:
                proposed=self._llm.choose_action(state,context)
                validated=self._accept_model(raw,proposed,plan)
                if validated:
                    # Score a complete accepted batch vs local; model must add utility.
                    model_utility=sum(self._order_utility(o,raw) for o in validated)
                    local_utility=sum(self._order_utility(o,raw) for o in plan)
                    if model_utility>local_utility+12:
                        plan=validated
                        notes=['AI economic/military plan']
                        self._last_was_llm=True
            except (ProviderError,ValueError,TypeError,KeyError) as exc:
                self._log('[riffo] Red Alert AI unavailable ('+type(exc).__name__+'); continuing economy plan')
        if not plan:
            return WAIT
        for o in plan:
            self._sent_at[self._order_key(o)]=self._clock()
        self._previous_orders=list(plan)
        self._last_order_key=None
        self._log('[riffo] Red Alert: '+', '.join(notes))
        return WithReasoning({'type':'orders','orders':plan}, '; '.join(notes)[:250])

    def on_action_result(self,result,context):
        # A rejected batch is never recorded in last_orders; retain reason and
        # back off, rather than trying the same expensive action every frame.
        if isinstance(result,dict) and (result.get('error')=='INVALID_ACTION' or result.get('accepted') is False):
            for o in self._previous_orders:
                self._rejected_until[self._order_key(o)]=self._clock()+32
            self._log('[riffo] Red Alert: server rejected batch; retrying after cooldown')
        self._previous_orders=[]
        if self._last_was_llm:
            self._llm.on_action_result(result,context)
        self._last_was_llm=False


def _types_for(species):
    name=_name(species)
    return tuple(_TYPES.get(name) or _MORE_SPECIES_TYPES.get(name) or ())


def _opponent_species(obs, my_seat=None):
    """Extract only explicitly identified opponent rosters, never the draft pool."""
    if not isinstance(obs,dict):
        return []
    names=[]
    def collect(obj):
        if isinstance(obj,str):
            names.append(obj)
        elif isinstance(obj,list):
            for item in obj:
                if isinstance(item,str): names.append(item)
                elif isinstance(item,dict):
                    name=item.get('species') or item.get('name')
                    if name: names.append(name)
        elif isinstance(obj,dict):
            if isinstance(obj.get('species'),str): names.append(obj['species'])
            for k in ('roster','team','picks','pokemon','cards','drafted'):
                if k in obj: collect(obj[k])
    for k in ('opponent_roster','opponent_team','opponent_picks','opponent_drafted',
              'opp_roster','enemy_roster','enemy_team'):
        collect(obs.get(k))
    for k in ('opponent','enemy'):
        if k in obs: collect(obs[k])
    for k in ('draft','draft_state','teams'):
        block=obs.get(k)
        if isinstance(block,dict):
            for k2 in ('opponent','enemy','opponent_roster','opponent_team'):
                collect(block.get(k2))
    # The draft state may use a public two-seat roster map. Select the OTHER
    # seat only if seat identity is explicit: never mistake our own roster for
    # the opponent's or treat remaining pool cards as drafted opponents.
    rosters=obs.get('rosters')
    if my_seat in (0,1):
        other=1-my_seat
        if isinstance(rosters,list) and len(rosters)>other:
            collect(rosters[other])
        elif isinstance(rosters,dict):
            for key in (str(other),'player_'+str(other),'seat_'+str(other)):
                if key in rosters:
                    collect(rosters[key]);break
    return list(dict.fromkeys(x for x in names if x))[:12]


def _counter_score(species, opponents):
    """Conservative type-only counter potential; neutral for unknown types."""
    attack=_types_for(species)
    if not attack or not opponents:
        return 0.0
    known=[x for x in opponents if _types_for(x)]
    if not known:
        return 0.0
    totals=[]
    for foe in known:
        # Penalize Pokémon that have a severe defensive weakness and cannot
        # threaten the enemy team. Account for both directions of matchup.
        offensive=max(_type_multiplier(t,foe) for t in attack)
        defensive=max(_type_multiplier(t,species) for t in _types_for(foe))
        totals.append(13*(offensive-1)-10*(defensive-1))
    totals.sort(reverse=True)
    return max(-25.0,min(35.0,sum(totals[:2])/max(1,len(totals[:2]))))


class CounterDraftPokemon(MatchupHybridPokemon):
    """Bring matchup-aware roster information into the AI's legal alternatives."""
    def choose_action(self, state, context):
        self._my_seat = getattr(context,'seat_position',None)
        return super().choose_action(state,context)

    def _alternatives(self,state):
        if not state.legal_actions:
            return []
        first=state.legal_actions[0]
        obs=state.raw.get('observation') if isinstance(state.raw,dict) else {}
        obs=obs if isinstance(obs,dict) else {}
        opp=_opponent_species(obs, getattr(self,'_my_seat',None))
        if first.action_id.startswith('draft_pick:'):
            # Generate ALL legal cards; don't discard a counter-pick before
            # its matchup value has been evaluated (old code kept top seven).
            pool={str(x.get('card_id')):x for x in obs.get('available_cards',[]) if isinstance(x,dict)}
            have=list(self.drafted_species)
            have_types=[_types_for(x) for x in have]
            options=[]
            for action in state.legal_actions:
                card=pool.get(action.action_id.partition(':')[2],{})
                species=card.get('species') or action.label or action.action_id
                types=_types_for(species)
                role=_name(species)
                utility=_species_score(species)+_counter_score(species,opp)
                if role in {'tornadus','whimsicott','murkrow'}:
                    utility+=12 if not any(_name(x) in {'tornadus','whimsicott','murkrow'} for x in have) else -15
                if role in {'incineroar','rillaboom','ironhands'}:
                    utility+=10 if not any(_name(x) in {'incineroar','rillaboom','ironhands'} for x in have) else -6
                if types and have_types:
                    utility-=4*sum(bool(set(types)&set(t)) for t in have_types)
                options.append({'action':action,'utility':utility,
                    'description':f'Draft {species}; counter score {_counter_score(species,opp):+.1f}; '
                                  f'opponent known={opp}; our roster={have}; card={_compact(card,350)}'})
            return sorted(options,key=lambda x:-x['utility'])[:9]
        if first.action_id=='select_lineup' and opp:
            template=first.input.get('action') or {}
            roster=list(dict.fromkeys(template.get('roster') or []))
            if len(roster)>=4:
                validator=lineup_choice(first,state)
                candidates=[]
                for bring in itertools.combinations(roster,4):
                    score=sum(_species_score(x)+_counter_score(x,opp) for x in bring)
                    score-=6*sum(1 for a,b in itertools.combinations(bring,2)
                                  if _types_for(a) and set(_types_for(a))==set(_types_for(b)))
                    for leads in itertools.combinations(bring,2):
                        utility=score+0.7*sum(_counter_score(x,opp) for x in leads)
                        if any(_name(x) in {'tornadus','whimsicott','murkrow'} for x in leads):utility+=9
                        try:
                            action=validator.build({'bring':list(bring),'leads':list(leads)})
                        except InvalidChoice:
                            continue
                        candidates.append({'action':action,'utility':utility,
                            'description':f'Bring {bring}; lead {leads}; opponents {opp}; '
                                          f'team counter utility {utility:.1f}'})
                if candidates:
                    return sorted(candidates,key=lambda c:-c['utility'])[:9]
        return super()._alternatives(state)


class RiffoAgent:
    """UCLA SDK entry point; one isolated instance per assigned game seat."""

    def __init__(self, provider=None, *, log=print):
        load_dotenv()
        self.provider = ResilientProvider(provider if provider is not None else provider_from_env())
        self.log = log
        self.pokemon = CounterDraftPokemon(self.provider, log=log)
        self.werewolf = HybridWerewolf(self.provider, log=log)
        self.red_alert = EconomyFirstRedAlert(self.provider, log=log)

    def choose_action(self, state, context):
        game_type = context.game_type or getattr(state, 'game_name', '')
        if game_type.startswith('pokemon'):
            return self.pokemon.choose_action(state, context)
        if game_type == 'werewolf':
            return self.werewolf.choose_action(state, context)
        if game_type == 'red_alert':
            return self.red_alert.choose_action(state, context)
        # Future game types fall back to the official multi-game agent.
        if not hasattr(self, '_generic'):
            self._generic = LLMAgent(self.provider, log=self.log)
        return self._generic.choose_action(state, context)

    def choose_message(self, state, context):
        if context.game_type == 'werewolf':
            return self.werewolf.choose_message(state, context)
        if not hasattr(self, '_generic'):
            self._generic = LLMAgent(self.provider, log=self.log)
        return self._generic.choose_message(state, context)

    def on_action_result(self, result, context):
        if context.game_type == 'red_alert':
            self.red_alert.on_action_result(result, context)
        elif hasattr(self, '_generic'):
            self._generic.on_action_result(result, context)


def create_agent():
    load_dotenv()
    return RiffoAgent()
