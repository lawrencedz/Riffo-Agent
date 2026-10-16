"""Riffo 7.0: grounded VGC, evidence-led Werewolf, AI-directed adaptive Red Alert.

Designed for the UCLA AltruAgent starter. The server's legal choices, private
seat observation and game configuration are authoritative. No embedded keys.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from altruagent import WAIT, WithReasoning, SendMessage, TERMINATE_MESSAGING
from examples.llm_agent import LLMAgent, _discrete_choice
from examples.llm.providers import ProviderError, provider_from_env
from examples.llm.base import InvalidChoice, object_schema
from agent.redalert_brain import AdaptiveRedAlert


def _dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _rows(v: Any) -> list[dict]:
    return [r for r in v if isinstance(r, dict)] if isinstance(v, list) else []


def _json(v: Any, limit: int = 24000) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(s) > limit:
        # Avoid silently replacing structured game facts with unrelated guesses.
        return s[:limit] + "...(state truncated; do not assume missing facts)"
    return s


def _seat(v: Any) -> int | None:
    if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 6:
        return v
    if isinstance(v, str):
        match = re.fullmatch(r"\s*(?:Player\s*)?([0-6])\s*", v, flags=re.I)
        if match:
            return int(match.group(1))
    if isinstance(v, dict):
        for field in ("position", "seat", "player", "id"):
            s = _seat(v.get(field))
            if s is not None:
                return s
    return None


def _role(observation: Any) -> str | None:
    if isinstance(observation, dict):
        for key in ("your_role", "my_role", "role"):
            value = observation.get(key)
            if isinstance(value, str) and value.strip().lower() in ("wolf", "werewolf", "seer", "villager"):
                return "wolf" if "wolf" in value.lower() else value.lower()
    if isinstance(observation, str):
        match = re.search(r"\b(?:your role\s*(?:is|:)|you are (?:a |the )?)\s*(werewolf|wolf|seer|villager)\b", observation, re.I)
        if match:
            return "wolf" if "wolf" in match[1].lower() else match[1].lower()
    return None


# This is a complete Gen-9 attacking-type chart, NOT a ranking of species or
# an attempt to simulate Showdown battles. It is used only to fact-check
# model-facing matchups, never to remove legal choices.
_TYPE_CHART = {
    "normal": (set(), {"rock", "steel"}, {"ghost"}),
    "fire": ({"grass", "ice", "bug", "steel"}, {"fire", "water", "rock", "dragon"}, set()),
    "water": ({"fire", "ground", "rock"}, {"water", "grass", "dragon"}, set()),
    "electric": ({"water", "flying"}, {"electric", "grass", "dragon"}, {"ground"}),
    "grass": ({"water", "ground", "rock"}, {"fire", "grass", "poison", "flying", "bug", "dragon", "steel"}, set()),
    "ice": ({"grass", "ground", "flying", "dragon"}, {"fire", "water", "ice", "steel"}, set()),
    "fighting": ({"normal", "ice", "rock", "dark", "steel"}, {"poison", "flying", "psychic", "bug", "fairy"}, {"ghost"}),
    "poison": ({"grass", "fairy"}, {"poison", "ground", "rock", "ghost"}, {"steel"}),
    "ground": ({"fire", "electric", "poison", "rock", "steel"}, {"grass", "bug"}, {"flying"}),
    "flying": ({"grass", "fighting", "bug"}, {"electric", "rock", "steel"}, set()),
    "psychic": ({"fighting", "poison"}, {"psychic", "steel"}, {"dark"}),
    "bug": ({"grass", "psychic", "dark"}, {"fire", "fighting", "poison", "flying", "ghost", "steel", "fairy"}, set()),
    "rock": ({"fire", "ice", "flying", "bug"}, {"fighting", "ground", "steel"}, set()),
    "ghost": ({"psychic", "ghost"}, {"dark"}, {"normal"}),
    "dragon": ({"dragon"}, {"steel"}, {"fairy"}),
    "dark": ({"psychic", "ghost"}, {"fighting", "dark", "fairy"}, set()),
    "steel": ({"ice", "rock", "fairy"}, {"fire", "water", "electric", "steel"}, set()),
    "fairy": ({"fighting", "dragon", "dark"}, {"fire", "poison", "steel"}, set()),
}


def _pokemon_id(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _type_multiplier(move_type: str, defending_types: list[str]) -> float:
    strong, resist, immune = _TYPE_CHART.get(str(move_type).lower(), (set(), set(), set()))
    multiplier = 1.0
    for t in defending_types:
        t = str(t).lower()
        if t in immune:
            return 0.0
        multiplier *= 2 if t in strong else (0.5 if t in resist else 1)
    return multiplier


class PokemonReference:
    """Optional local Showdown metadata; no network requests during a match.

    Run scripts/setup_pokemon_reference.py ONCE on the machine running Riffo
    to get the public Pokedex/move data. If missing, play normally without it.
    """
    def __init__(self):
        self.species, self.moves = {}, {}
        try:
            file = Path(__file__).with_name('pokemon_reference.json')
            if file.is_file():
                data = json.loads(file.read_text(encoding='utf-8'))
                self.species = _dict(data.get('species'))
                self.moves = _dict(data.get('moves'))
        except (OSError, ValueError, TypeError):
            pass

    def brief(self, observation: Any, options: Any) -> dict:
        """Extract only Pokémon and moves observed by this player."""
        species, moves = set(), set()

        def scan(obj: Any, depth: int = 0):
            if depth > 9:
                return
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key in ('species', 'baseSpecies') and isinstance(val, str):
                        species.add(_pokemon_id(val))
                    if key == 'move_id' and isinstance(val, str):
                        moves.add(_pokemon_id(val))
                    if key == 'is' and isinstance(val, str) and val.startswith('OPPONENT '):
                        species.add(_pokemon_id(val[len('OPPONENT '):]))
                    if key in ('roster', 'bring', 'leads', 'opponent_roster', 'picks', 'drafted', 'team') and isinstance(val, list):
                        for x in val:
                            if isinstance(x, str):
                                species.add(_pokemon_id(x))
                    scan(val, depth + 1)
            elif isinstance(obj, list):
                for x in obj[:90]:
                    scan(x, depth + 1)
        scan(observation)
        scan(options)
        brief = {
            'species': {key: self.species[key] for key in sorted(species) if key in self.species},
            'moves': {key: self.moves[key] for key in sorted(moves) if key in self.moves},
        }
        # Exact move-vs-target matchups derived from the full type chart and
        # Showdown species data; they are FACTS for the AI, not local rankings.
        annotations = []
        for slot in _dict(options).get('slots') or []:
            for opt in _dict(slot).get('options') or []:
                move_id = _pokemon_id(_dict(opt).get('move_id'))
                move = self.moves.get(move_id, {})
                if _dict(opt).get('type') != 'move' or not move.get('power'):
                    continue
                for target in _dict(opt).get('targets') or []:
                    if not isinstance(target, dict):
                        continue
                    described = str(target.get('is', ''))
                    if not described.startswith('OPPONENT '):
                        continue
                    who = _pokemon_id(described[len('OPPONENT '):])
                    foe = self.species.get(who)
                    if not foe:
                        continue
                    factor = _type_multiplier(move.get('type', ''), foe.get('types', []))
                    annotations.append({'move':move_id, 'opponent':who,
                                        'target':target.get('target'), 'type_multiplier':factor})
        if annotations:
            brief['visible_move_matchups_BASE_TYPES_ONLY'] = annotations[:70]
        return brief


class RiffoAgent(LLMAgent):
    """Expert-model decision maker using official schema-bound legal choices."""

    def __init__(self, provider=None, *, log=print, clock=time.monotonic):
        self.log = log
        self.clock = clock
        self.own_moves: list[dict] = []
        self.pokemon_reference = PokemonReference()
        self._opening_review_done = False
        # Store active species across battle decisions to flag Fake Out misuse.
        self._active_field_turns: dict[int, tuple[str, int]] = {}
        self.wolf_teammates: set[int] = set()
        self._known_role: str | None = None
        super().__init__(provider or provider_from_env(),
                         realtime_players={"red_alert": AdaptiveRedAlert},
                         log=log, clock=clock)

    def _battle_warnings(self, prompt: dict) -> dict:
        """Mechanics facts for the LLM, not a handcrafted move scorer.

        The official adapter supplies current active species and legal moves.
        Counting uninterrupted appearances avoids repeatedly using Fake Out,
        which only flinches on a Pokemon's first turn on the field.
        """
        current: dict[int, tuple[str, int]] = {}
        warnings: list[str] = []
        slots = prompt.get('slots', []) if isinstance(prompt, dict) else []
        recent = self.own_moves[-1] if self.own_moves else {}
        previous = _dict(recent.get('action')) if recent.get('phase') == 'moving' else {}
        for row in slots if isinstance(slots, list) else []:
            if not isinstance(row, dict):
                continue
            slot = row.get('slot')
            if type(slot) is not int or slot not in (0, 1):
                continue
            active = row.get('active')
            if isinstance(active, dict):
                active = active.get('species') or active.get('name')
            species = _pokemon_id(active)
            if not species:
                # Forced replacement choices often describe an empty slot.
                # Do not carry forward a stale species across a faint or switch.
                continue
            old = self._active_field_turns.get(slot)
            turns = old[1] + 1 if old and old[0] == species else 1
            current[slot] = (species, turns)
            offered = [_pokemon_id(opt.get('move_id')) for opt in row.get('options', [])
                       if isinstance(opt, dict) and opt.get('type') == 'move']
            if 'fakeout' in offered and turns > 1:
                warnings.append(f'Slot {slot} ({active}) has remained active for at least {turns} decisions: '
                                'Fake Out will fail unless it left and re-entered; pick a different move.')
            last = _dict(previous.get(f'slot_{slot}'))
            if turns > 1 and last.get('type') == 'move' and _pokemon_id(last.get('move_id')) in ('protect', 'detect'):
                if any(m in offered for m in ('protect', 'detect')):
                    warnings.append(f'Slot {slot} ({active}) used Protect/Detect on its preceding action. '
                                    'Consecutive Protect success is approximately 1/3 on the second attempt, '
                                    'and drops further on additional consecutive attempts.')
        self._active_field_turns = current
        return {
            'observed_active_field_turns': {str(k): {'species': v[0], 'consecutive_decisions_seen': v[1]}
                                           for k, v in current.items()},
            'strategic_warnings': warnings,
            'scope': 'These facts concern only currently observed active Pokemon; unknown switches/abilities/items '
                     'must not be inferred. All server-supplied moves remain selectable.'
        }

    def _critical_battle_warning(self, action: dict) -> str | None:
        """Check only two universal turn-to-turn mechanics after model choice.

        All other strategic decisions belong to the model; this does not
        implement a move database or score alternatives.
        """
        if not isinstance(action, dict) or action.get('type') != 'doubles_turn':
            return None
        previous = self.own_moves[-1] if self.own_moves else {}
        last_action = _dict(previous.get('action')) if previous.get('phase') == 'moving' else {}
        errors = []
        for slot in (0, 1):
            choice = _dict(action.get(f'slot_{slot}'))
            if choice.get('type') != 'move':
                continue
            move = _pokemon_id(choice.get('move_id'))
            current = self._active_field_turns.get(slot)
            if current is None or current[1] <= 1:
                continue
            if move == 'fakeout':
                errors.append(f'slot_{slot} {current[0]} Fake Out will fail; it did not just enter the field')
            last = _dict(last_action.get(f'slot_{slot}'))
            if move in ('protect', 'detect') and last.get('type') == 'move' and \
                    _pokemon_id(last.get('move_id')) in ('protect', 'detect'):
                errors.append(f'slot_{slot} {current[0]} is attempting consecutive Protect/Detect '
                              'with about 1/3 chance on the second attempt')
        return '; '.join(errors) if errors else None

    def _wolf_info(self, state: Any, context: Any) -> tuple[str | None, int | None, set[int]]:
        raw = _dict(getattr(state, "raw", {}))
        obs = raw.get("observation", getattr(state, "observation", None))
        me = _seat(raw.get("your_position"))
        if me is None:
            me = _seat(getattr(context, "seat_position", None))
        role = _role(obs) or self._known_role
        if role:
            self._known_role = role
        if role != "wolf":
            return role, me, set()
        if isinstance(obs, dict):
            for key in ("wolf_teammate", "wolf_partner", "other_wolf", "other_wolves",
                        "teammate_position", "teammates", "wolf_team", "wolves"):
                val = obs.get(key)
                if isinstance(val, dict):
                    val = next((val[k] for k in ("members", "players", "positions", "seats") if k in val), val)
                sources = val if isinstance(val, list) else [val]
                for member in sources:
                    who = _seat(member)
                    if who is not None and who != me:
                        self.wolf_teammates.add(who)
        if isinstance(obs, str):
            for m in re.finditer(r"(?:other wolf|wolf partner|wolf teammate|fellow wolf)\s*(?:is|:)?\s*Player\s*([0-6])", obs, re.I):
                who = int(m[1])
                if who != me:
                    self.wolf_teammates.add(who)
        gs = _dict(raw.get("game_state"))
        if gs.get("phase") == "night_wolf" and me is not None:
            offered = {_seat(getattr(a, "action_id", None)) for a in getattr(state, "legal_actions", [])}
            offered.discard(None)
            dead = _rows(gs.get("dead"))
            if not dead and len(offered) == 5:
                # With 2 wolves + 5 villagers, one excluded living seat other
                # than ourselves is our packmate, based on *our own* legal list.
                missing = set(range(7)) - offered - {me}
                if len(missing) == 1:
                    self.wolf_teammates |= missing
        return role, me, set(self.wolf_teammates)

    @staticmethod
    def _private_checks(observation: Any) -> dict[int, str]:
        """Only a seer's own private observation can certify investigations."""
        if not isinstance(observation, dict) or _role(observation) != "seer":
            return {}
        answers: dict[int, str] = {}
        for field in ("seer_results", "investigations", "investigation_results"):
            value = observation.get(field)
            if isinstance(value, dict):
                rows = [{"player": k, "result": v} for k, v in value.items()]
            else:
                rows = _rows(value)
            for row in rows:
                who = _seat(row.get("player", row.get("target", row.get("position"))))
                result = row.get("result", row.get("role", ""))
                if isinstance(result, dict):
                    result = result.get("role", result.get("result", ""))
                result = str(result).lower().strip()
                if who is not None and result in ("wolf", "werewolf", "villager", "seer"):
                    answers[who] = "wolf" if "wolf" in result else "villager"
        return answers

    def choose_action(self, state: Any, context: Any):
        kind = str(getattr(context, "game_type", "") or "")
        if kind == "werewolf":
            self._remember(state)
            role, me, teammates = self._wolf_info(state, context)
            gs = _dict(_dict(getattr(state, "raw", {})).get("game_state"))
            candidates = list(state.legal_actions)
            if role == "wolf":
                safe = [a for a in candidates if _seat(a.action_id) not in teammates]
                if safe:
                    candidates = safe
                elif teammates:
                    self.log("[riffo6] Wolf: no safe candidate; retaining server legal list")
                elif gs.get("phase") == "day_vote":
                    self.log("[riffo6] Wolf teammate not identified; voting with incomplete private information")
            elif role == "seer" and gs.get("phase") == "day_vote":
                checks = self._private_checks(_dict(getattr(state, "raw", {})).get("observation"))
                confirmed_wolves = [a for a in candidates if checks.get(_seat(a.action_id)) == "wolf"]
                if confirmed_wolves:
                    candidates = confirmed_wolves
                else:
                    permitted = [a for a in candidates if checks.get(_seat(a.action_id)) != "villager"]
                    if permitted:
                        candidates = permitted
            if not candidates:
                return WAIT
            choice = _discrete_choice(candidates)
            value, answer = self._ask(choice, state, context, counter="decision_calls")
            if value is None:
                value = choice.fallback()
            summary = str(_dict(answer).get("reasoning_summary") or "Evidence-based Werewolf choice")[:220]
            self.log(f"[riffo6] Werewolf {gs.get('phase', 'move')}: {getattr(value,'action_id',value)}")
            return WithReasoning(value, summary)
        result = super().choose_action(state, context)
        if kind.startswith("pokemon"):
            a = result.action if isinstance(result, WithReasoning) else result
            self.own_moves.append({"phase": getattr(state, "phase", None), "action":
                                   a if isinstance(a, dict) else getattr(a, "action_id", str(a))})
            self.own_moves = self.own_moves[-7:]
        return result

    def choose_message(self, state: Any, context: Any):
        if getattr(context, "game_type", "") != "werewolf":
            return super().choose_message(state, context)
        role, me, pack = self._wolf_info(state, context)
        sent = super().choose_message(state, context)
        if not isinstance(sent, SendMessage):
            return sent
        msg = sent.content
        public = not getattr(sent, "recipients", [])
        # Guard only obviously fabricated private facts or accidental public
        # self-identification. General suspicions and rhetorical argument remain AI-led.
        if public and role != "seer" and re.search(r"\b(I (?:am|was) (?:the )?seer|I investigated|my seer result)\b", msg, re.I):
            self.log("[riffo6] Suppressed ungrounded public seer claim")
            return TERMINATE_MESSAGING
        if public and role == "wolf" and re.search(r"\b(I am (?:a )?(?:wolf|werewolf)|we are wolves)\b", msg, re.I):
            self.log("[riffo6] Suppressed accidental public wolf reveal")
            return TERMINATE_MESSAGING
        if public and role == "wolf" and any(re.search(rf"\b(?:vote\s+(?:for|on|against)|lynch|eliminate)\s+Player\s*{t}\b", msg, re.I) for t in pack):
            self.log("[riffo6] Suppressed vote request against wolf teammate")
            return TERMINATE_MESSAGING
        return sent

    def _ask(self, choice: Any, state: Any, context: Any, *, counter: str):
        kind = str(getattr(context, "game_type", "") or "")
        if not kind.startswith("pokemon") and kind != "werewolf":
            return super()._ask(choice, state, context, counter=counter)
        raw = _dict(getattr(state, "raw", {}))
        phase = getattr(state, "phase", "")
        base = {
            "game": kind, "your_seat": getattr(context, "seat_position", None),
            "phase": phase, "decision": choice.kind,
            "legal_options_and_instructions": choice.prompt,
            "your_observation": raw.get("observation"),
        }
        if kind == "werewolf":
            role, me, pack = self._wolf_info(state, context)
            base.update({
                "private_role": role, "your_player_number": me,
                "known_wolf_teammates_PRIVATE": sorted(pack) if role == "wolf" else [],
                "public_game_state": raw.get("game_state"),
                "players": raw.get("players"),
                "public_discussion": self._transcript[-28:],
                "WARNING": "Names may be duplicated across seats. Trust Player numbers and explicit private facts. "
                           "Previous public votes do not reveal current hidden roles."
            })
            system = (
                "You are playing simplified seven-person Werewolf: 2 wolves, 1 seer, 4 villagers. "
                "Maximize your side's chance to win using evidence and incentives, not fixed scripts. "
                "Analyze actual votes, revealed dead roles, contradictions, seer results and role incentives. "
                "Public chat is NOT verified evidence; opponents may lie. "
                "If you are a wolf, protect your known partner, coordinate privately when valuable, and seek parity. "
                "If you are the seer, use only your OWN private verified investigations as certain. "
                "If you are a villager, reason from public evidence and coalition incentives. "
                "Night 1 is peaceful, daytime plurality eliminates; 3 consecutive no-elimination days lose for villagers. "
                "Votes for fellow wolves are harmful. In chat, cite concrete public evidence when available; "
                "on Day 1 with no evidence, express uncertainty rather than fabricate witnessed actions. "
                "Keep chat under 40 words. Never claim a fake seer investigation. "
                "Choose ONLY a supplied legal action or discussion recipient. "
                "Answer the exact required strict JSON schema; public reasoning_summary must not leak private roles."
            )
            limit, budget = 15000, 16 if choice.kind == "message" else 15
        else:
            if choice.kind == 'doubles_turn':
                base['battle_mechanics'] = self._battle_warnings(choice.prompt)
            reference = self.pokemon_reference.brief(raw.get("observation"), choice.prompt)
            if reference["species"] or reference["moves"]:
                base["verified_pokemon_reference"] = reference
            base.update({
                "public_game_information": {k: raw.get(k) for k in (
                    "game_state", "time", "match_rules", "players", "field") if k in raw},
                "our_recent_actions": self.own_moves[-6:],
                "recent_events": self._transcript[-8:],
            })
            system = (
                "You are an expert competitive Pokémon VGC doubles coach. You control one seat of a "
                "snake draft, 4-of-6 preview, simultaneous two-Pokémon battle. "
                "Reason from CURRENT legal choices and visible state, NOT from a short hardcoded rankings table. "
                "During draft, build a synergistic team with speed control, coverage, priority, defense, "
                "support, and responses to publicly observed opposing picks. Only assume actual known moves/items. "
                "At preview, choose 4 and leads specifically to beat the opposing roster, including probable leads. "
                "At battle, prioritize NOT LOSING two Pokemon in a turn: consider whether the opposing "
                "leads can KO ours, and whether Protect, Fake Out or a switch is worth more than attacking. "
                "Use verified_pokemon_reference for base typings and move power when present; "
                "possible abilities are not confirmed, and Tera or temporary transformations override base types. If absent, "
                "carefully check both types and never mistake 0.25x for neutral. "
                "Fake Out only works on the user's first turn after entering the field; if that Pokemon "
                "stayed in on previous turns, choosing Fake Out again wastes its action. "
                "A second consecutive Protect/Detect has only about a 1/3 success probability; "
                "never assume it works twice in a row. Check battle_mechanics warnings. "
                "Do not automatically double Fake Out: compare damage and Protect when both opening "
                "Pokemon are threatened; targets may avoid flinch because of items, abilities or field effects. "
                "Consider opposing Trick Room and Tailwind modes at preview; offensive speed alone "
                "is insufficient against Trick Room. Do not claim a type immunity if merely resisted. "
                "At battle, reason about BOTH Pokémon jointly: Protect and switching, speed control, "
                "likely opponent Protect/switch, type immunities, status, stat stages, HP, expected KOs, "
                "field conditions, move recoil and self-debuffs. Do not spam a move into repeated Protect. "
                "Do not automatically double-target if it overkills or leaves another threat unchecked. "
                "Respect targets: +1/+2 opponents, -1/-2 your own side, 0 no target. "
                "Prefer a solid legal move over a slow or speculative calculation. "
                "Choose ONLY provided indexes/roster species and legal targets, and answer strict JSON. "
                "Keep public reasoning_summary short and avoid claiming unknown events."
            )
            limit = 29000
            budget = 9 if choice.kind == "choose_action" and phase == "draft" else (22 if choice.kind == "doubles_turn" else 24)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _json(base, limit)},
        ]
        deadline = self.clock() + budget
        for attempt in range(2):
            left = deadline - self.clock()
            if left < 2.5:
                break
            setattr(self, counter, getattr(self, counter) + 1)
            try:
                answer = self._provider.complete_structured(
                    messages, choice.kind, choice.schema,
                    timeout=min(16.0, max(2.5, left - 0.7)))
                value = choice.build(answer)
                if kind.startswith('pokemon') and choice.kind == 'doubles_turn':
                    obvious = self._critical_battle_warning(value)
                    if obvious and deadline - self.clock() > 5.5:
                        # Model-first correction, not a hardcoded replacement move.
                        try:
                            self.log('[riffo6.3] Tactical rule check: ' + obvious)
                            fixed = self._provider.complete_structured([
                                {'role': 'system', 'content': (
                                    'Correct the VGC action using only the supplied legal options. '
                                    'Fake Out cannot work after the Pokemon stayed in since a previous turn; '
                                    'consecutive Protect is unreliable. Choose the best OTHER joint action '
                                    'considering opponent targets, HP, moves, and field. Use the SAME schema.')},
                                {'role': 'user', 'content': _json({
                                    'mechanical_problem': obvious,
                                    'legal_options_and_instructions': choice.prompt,
                                    'observation': raw.get('observation'),
                                    'original_selection': answer,
                                    'battle_mechanics': base.get('battle_mechanics', {}),
                                }, 18000)},
                            ], choice.kind, choice.schema,
                               timeout=min(4.5, deadline - self.clock() - 1.0))
                            revised = choice.build(fixed)
                            if not self._critical_battle_warning(revised):
                                value, answer = revised, fixed
                                self.log('[riffo6.3] Corrected risky battle action')
                        except (ProviderError, InvalidChoice, ValueError, TypeError, KeyError, TimeoutError):
                            self.log('[riffo6.3] Tactical correction unavailable; using validated original')
                # Independent *model* critique, never a locally ranked shortlist.
                # Only the preview and opening battle decision are reviewed to
                # control API costs and protect the 420-second battle time bank.
                if kind.startswith('pokemon'):
                    opening = choice.kind == 'doubles_turn' and not self._opening_review_done
                    reviewing = choice.kind == 'select_lineup' or opening
                    if opening:
                        self._opening_review_done = True
                    if reviewing and deadline - self.clock() >= 8:
                        review_schema = choice.schema
                        review_input = {
                            'phase': phase, 'legal_options_and_instructions': choice.prompt,
                            'player_observation': raw.get('observation'),
                            'verified_pokemon_reference': base.get('verified_pokemon_reference', {}),
                            'first_decision': answer,
                        }
                        try:
                            revision = self._provider.complete_structured([
                                {'role':'system','content':(
                                    'You are the independent VGC tactical reviewer. The first coach chose an action. '
                                    'Before approving, check all probable opponent opening attacks, double KOs, '
                                    'speed control, move typing, immunities, Protect and favorable switches. '
                                    'In team preview consider which leads survive the probable opponent leads. '
                                    'Correct a risky selection when justified, otherwise repeat the exact selection. '
                                    'Return the SAME required schema; only server-offered legal choices are allowed.')},
                                {'role':'user','content':_json(review_input, 25000)},
                            ], choice.kind, review_schema,
                            timeout=min(8.0, deadline-self.clock()-1.5))
                            checked = choice.build(revision)
                            if checked != value:
                                self.log('[riffo6.1] Pokemon tactical review revised the initial decision')
                            value, answer = checked, revision
                        except (ProviderError, InvalidChoice, ValueError, TypeError, KeyError, TimeoutError):
                            self.log('[riffo6.1] Pokemon tactical review unavailable; keeping valid original decision')
                return value, answer
            except (ProviderError, InvalidChoice, ValueError, TypeError, KeyError, TimeoutError) as exc:
                self.log(f"[riffo6] {kind} {choice.kind} model/validation failed: {type(exc).__name__}")
                messages.append({"role": "user", "content": "Invalid prior response. Reconsider only the supplied legal options and follow the schema exactly."})
        return None, None


def create_agent():
    load_dotenv()
    return RiffoAgent()
