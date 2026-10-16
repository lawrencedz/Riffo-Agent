"""Riffo 7.0 -- AI-directed Red Alert controller.

The LLM controls economy, production, expansion, tactics and attack timing.
This module is only a low-latency I/O scheduler and safety envelope:
  - deploy an idle MCV / place a completed building without waiting on network;
  - rescue idle harvesters or a demonstrably stalled economy;
  - supply current/previous observations and real order feedback to the model;
  - validate the model's orders against the live server schema and observation;
  - never block the real-time loop waiting for a model response.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import Counter, deque
from typing import Any

from altruagent import WAIT, WithReasoning
from examples.llm.providers import ProviderError
from examples.llm.redalert import (
    _cell, _limits, answer_schema, build_system_prompt, build_user_prompt,
    order_fields, validate_orders,
)


def obj(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def rows(v: Any) -> list[dict]:
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def kind(v: dict) -> str:
    return str(v.get("type") or "").lower()


def integer(v: Any) -> bool:
    return type(v) is int


def short(v: Any, length: int = 1250) -> str:
    return json.dumps(v, default=str, separators=(",", ":"), ensure_ascii=False)[:length]


class AdaptiveRedAlert:
    """One real-time match, one asynchronous LLM decision at a time."""

    WORKER_TIMEOUT = 8.0
    NONCOMBAT = frozenset({"mcv", "harv", "harvester", "miner", "truk", "truck"})
    BUILDINGS = frozenset({"powr", "apwr", "barr", "tent", "proc", "weap", "afld", "silo",
                           "dome", "stek", "atek", "hpad", "fix", "gun", "ftur", "tsla",
                           "sam", "agun", "mslo", "iron", "pdox", "hbox", "pbox", "sbag",
                           "brik", "syrd", "medi"})
    PRICES = {"powr": 300, "apwr": 500, "barr": 300, "tent": 300,
              "proc": 2000, "weap": 2000, "harv": 1400, "e1": 100, "e3": 300}

    def __init__(self, provider: Any, *, log=print, clock=time.monotonic):
        self.provider, self.log, self.clock = provider, log, clock
        self.max_calls = max(0, min(1000, int(os.getenv("RIFFO_RTS_AI_CALLS", "400"))))
        self.timeout = max(2.0, min(25.0, float(os.getenv("RIFFO_RTS_TIMEOUT", "8"))))
        self.ai_calls = 0
        self.last_call = -1e9
        self.next_call = -1e9
        self.worker_active = False
        self.generation = 0
        self.replies: queue.SimpleQueue = queue.SimpleQueue()
        self.last_reason = "No AI decision yet"
        self.started = self.clock()
        self.last_status = self.clock()
        self.sent_recent: dict[str, float] = {}
        self.refused_recent: dict[str, float] = {}
        self.last_batch: list[dict] = []
        self.last_orders_token: str | None = None
        self.last_problems_token: str | None = None
        self.last_results: deque[str] = deque(maxlen=9)
        self.snapshots: deque[dict] = deque(maxlen=16)
        self.last_snapshot = -1e9
        self.seen_enemy_types: Counter = Counter()
        self.enemy_buildings_seen: set[str] = set()
        self.last_cash_increase = self.clock()
        self.cash_previous: float | None = None
        self.last_harvest_rescue: dict[int, float] = {}
        self.errors = 0
        self.last_signature: str | None = None
        self.last_request_signature: str | None = None
        self.empty_answers = 0

    def _price(self, item: str, context: Any) -> int | None:
        rules = obj(obj(getattr(context, "game_config", {})).get("rules"))
        for key in ("buildings", "units"):
            found = obj(obj(rules.get(key)).get(item)).get("cost")
            if type(found) in (float, int) and 0 <= found <= 100000:
                return int(found)
        return self.PRICES.get(item)

    def _observe(self, raw: dict, obs: dict) -> None:
        now = self.clock()
        cash = obj(obs.get("economy")).get("cash")
        if isinstance(cash, (int, float)) and not isinstance(cash, bool):
            if self.cash_previous is None or cash > self.cash_previous + 1:
                self.last_cash_increase = now
            self.cash_previous = float(cash)
        if now - self.last_snapshot < 5.0:
            return
        self.last_snapshot = now
        ours = rows(obs.get("units"))
        enemies = rows(obs.get("visible_enemies"))
        enemy_buildings = rows(obs.get("visible_enemy_buildings"))
        self.seen_enemy_types.update({kind(e): 1 for e in enemies if kind(e)})
        self.enemy_buildings_seen.update(kind(e) for e in enemy_buildings if kind(e))
        self.snapshots.append({
            "seconds": round(now - self.started, 1),
            "cash": cash,
            "our_units": sum(kind(u) not in self.NONCOMBAT and not u.get("wreck") for u in ours),
            "our_buildings": len(rows(obs.get("buildings"))),
            "visible_enemy_units": len(enemies),
            "visible_enemy_buildings": len(enemy_buildings),
            "harvesters": sum(kind(u) in ("harv", "harvester", "miner") for u in ours),
        })

    def _feedback(self, raw: dict):
        last = obj(raw.get("last_orders"))
        if last and not last.get("pending"):
            token = short(last, 2500)
            if token != self.last_orders_token:
                self.last_orders_token = token
                for result in rows(last.get("results")):
                    status = str(result.get("status", "")).lower()
                    reason = str(result.get("reason") or "")
                    if (status not in ("", "accepted", "ok", "success", "done")
                            or reason.lower() not in ("", "ok", "accepted", "success")):
                        entry = f"{result.get('cmd')}: {status}: {reason}"[:200]
                        self.last_results.append(entry)
                        self.log("[riffo7] SERVER: " + entry)
        problems = rows(raw.get("recent_order_problems"))
        if problems:
            marker = short(problems[-3:], 1000)
            if marker != self.last_problems_token:
                self.last_problems_token = marker
                for p in problems[-3:]:
                    item = f"{p.get('cmd')}: {p.get('reason')}: {p.get('detail')}"[:220]
                    self.last_results.append(item)
                self.log("[riffo7] recent rejected orders: " + marker)

    def _snapshot(self, raw: dict, obs: dict, context: Any) -> dict:
        own = rows(obs.get("units"))
        enemies = rows(obs.get("visible_enemies"))
        buildings = rows(obs.get("buildings"))
        visible_buildings = rows(obs.get("visible_enemy_buildings"))
        econ = obj(obs.get("economy"))
        now = self.clock()
        stale = max(0, now - self.last_cash_increase)
        history = list(self.snapshots)
        dynamic = {
            "income_warning": "No observed cash INCREASE for %d seconds; cash may be flat while harvesting or stuck." % stale if stale >= 22 else None,
            "observed_opponent_not_omniscient": {
                "enemy_units_visible_now": dict(Counter(kind(u) for u in enemies if kind(u))),
                "enemy_buildings_visible_now": dict(Counter(kind(b) for b in visible_buildings if kind(b))),
                "enemy_types_seen_at_least_once": sorted(self.seen_enemy_types),
                "enemy_building_types_seen_at_least_once": sorted(self.enemy_buildings_seen),
            },
            "our_units_now": dict(Counter(kind(u) for u in own if kind(u))),
            "our_buildings_now": dict(Counter(kind(b) for b in buildings if kind(b))),
            "cash": econ.get("cash"),
            "power_supplied": econ.get("power_provided"),
            "power_drain": econ.get("power_drained"),
            "harvesters": [{"id":u.get("id"), "idle":u.get("idle"), "cell":_cell(u),
                            "activity":u.get("order", u.get("activity"))}
                           for u in own if kind(u) in ("harv", "harvester", "miner")][:10],
            "recent_snapshots_5s_apart": history[-10:],
            "recent_order_refusals": list(self.last_results)[-7:],
            "last_command_plan": self.last_reason,
            "safety_actions_already_automatic": ["deploy MCV", "place completed structures", "wake idle or very-stalled harvester"],
        }
        return dynamic

    def _worker(self, generation: int, messages: list[dict], schema: dict, started: float) -> None:
        try:
            reply = self.provider.complete_structured(messages, "riffo7_redalert_orders", schema, timeout=self.timeout)
            if not isinstance(reply, dict):
                raise ValueError("model response was not an object")
            self.replies.put((generation, started, reply, None))
        except (ProviderError, ValueError, TypeError, TimeoutError, OSError) as exc:
            # ProviderError reports safe status codes without exposing keys.
            self.replies.put((generation, started, None, f"{type(exc).__name__}: {str(exc)[:140]}"))
        except Exception as exc:
            self.replies.put((generation, started, None, type(exc).__name__))

    def _interval(self, obs: dict, raw: dict) -> float:
        near = self._threats(obs)
        t = obj(raw.get("time")).get("game_seconds")
        if near:
            return 2.8
        if isinstance(t, (int, float)) and t < 100:
            return 3.0
        return 5.0

    def _threats(self, obs: dict) -> list[dict]:
        center = _cell(obs.get("base_center"))
        if center is None:
            return []
        return [e for e in rows(obs.get("visible_enemies")) if _cell(e) is not None
                and sum((x-y)**2 for x,y in zip(center, _cell(e))) <= 18**2]

    def _request(self, raw: dict, obs: dict, context: Any):
        now = self.clock()
        if self.worker_active and now - self.last_call > self.timeout + 5:
            # A provider that never returns must not disable autonomous RTS play.
            self.worker_active = False
            self.generation += 1  # reject any late answer from the stuck call
            self.log("[riffo7] AI request watchdog expired; starting fresh")
        if self.worker_active or self.ai_calls >= self.max_calls:
            return
        # Changes in visible opponents, tech, or queues warrant a quick
        # reevaluation even if our ordinary interval hasn't elapsed.
        signature = short({
            "buildings": sorted(kind(b) for b in rows(obs.get("buildings"))),
            "queue": sorted(str(p.get("item")) for p in rows(obs.get("production"))),
            "enemy_units": sorted(kind(e) for e in rows(obs.get("visible_enemies"))),
            "enemy_buildings": sorted(kind(b) for b in rows(obs.get("visible_enemy_buildings"))),
            "cash_band": int(float(obj(obs.get("economy")).get("cash") or 0) // 500),
        }, 1500)
        changed = signature != self.last_request_signature
        if now < self.next_call and not (changed and now - self.last_call >= 2.0):
            return
        config = obj(getattr(context, "game_config", {}))
        schema = answer_schema(order_fields(config))
        base = build_user_prompt(raw, None, config, max_chars=10500)
        information = self._snapshot(raw, obs, context)
        instructions = (
            "YOU are the real-time Red Alert commander. You, not Python, choose the economy, build order, "
            "army composition, expansion, scouting, defense, attack timing, and unit micro. "
            "Each answer is a batch of actual game commands, not a high-level stance. "
            "Use CURRENT available_production and actual actor ids and map cells; don't invent units or enemy knowledge. "
            "Compare our live growth and military to the VISIBLE opponent and adapt aggressively to what they show: "
            "if behind in units/economy, invest in scalable production or counter-units; if their raid approaches, "
            "defend; if their base is vulnerable, attack/scout with an appropriate force. Enemy units outside "
            "vision are UNKNOWN, not absent. Use the opponent history supplied as your best evidence. "
            "ECONOMIC FAILURE TO AVOID: early infantry spending previously bankrupted our opening while "
            "refinery/war-factory tech was unaffordable. Preserve enough funds for proc and weap until "
            "each is built/queued; do not repeatedly train troops if that would prevent economy scaling. "
            "You are free to change that plan when actual enemy threats justify it; explain the tradeoff. "
            "Refineries come with harvesters; inspect harvester work and order harvest for stuck/idle trucks. "
            "Buildings being constructed are already in production: do not repeatedly queue the same building. "
            "When cash is too low, choose useful movement/defense/scouting/repair orders or an empty batch "
            "rather than unaffordable training. You MAY try queueing a strategically essential BUILD even if "
            "cash is below full price: the server will decide whether progressive-payment queues are allowed; "
            "read the resulting acceptance/refusal and don't spam a refused command. "
            "Finish an enemy by destroying all buildings/MCV, not just units. "
            "The world runs in real time and you will be asked again in a few seconds. Respond in strict JSON."
        )
        messages = [
            {"role": "system", "content": build_system_prompt(config) + "\n\n" + instructions},
            {"role": "user", "content": base + "\n\nADAPTIVE TRENDS AND FEEDBACK:\n" + short(information, 4800)},
        ]
        self.generation += 1
        gen = self.generation
        self.worker_active = True
        self.ai_calls += 1
        self.last_call = now
        self.last_request_signature = signature
        self.next_call = now + self._interval(obs, raw)
        threading.Thread(target=self._worker, args=(gen, messages, schema, now), daemon=True).start()

    @staticmethod
    def _order_key(order: dict) -> str:
        # Stable cooldown across different AI calls but not different targets/groups.
        return short(order, 550)

    def _safe_orders(self, proposed: list, raw: dict, context: Any) -> tuple[list[dict], int]:
        obs = obj(raw.get("observation"))
        config = obj(getattr(context, "game_config", {}))
        valid, bad = validate_orders(proposed, order_fields(config), _limits(raw))
        units = {u["id"]: u for u in rows(obs.get("units")) if integer(u.get("id")) and not u.get("wreck")}
        own_buildings = {b["id"] for b in rows(obs.get("buildings")) if integer(b.get("id"))}
        enemies = {e["id"] for e in (rows(obs.get("visible_enemies")) +
                                      rows(obs.get("visible_enemy_buildings"))) if integer(e.get("id"))}
        ready = {str(p.get("item", "")).lower() for p in rows(obs.get("production")) if p.get("ready") is True}
        queued = Counter(str(p.get("item", "")).lower() for p in rows(obs.get("production")))
        available = {x.lower() for x in obs.get("available_production") or [] if isinstance(x, str)}
        bounds = obj(raw.get("map")).get("bounds")
        cash = obj(obs.get("economy")).get("cash")
        budget = float(cash) if type(cash) in (int, float) else 0.0
        under_attack = bool(self._threats(obs))
        built_types = {kind(b) for b in rows(obs.get("buildings"))}
        proc_built = "proc" in built_types
        weap_built = "weap" in built_types
        # Prices can be charged progressively while a building constructs.
        # Keep enough *uncommitted* cash for both core tech steps. The old bot
        # treated 'proc queued' as 'proc already fully paid' and overspent the
        # remainder on infantry, permanently trapping its war factory.
        tech_reserve = 0
        if not proc_built:
            tech_reserve += self._price("proc", context) or 2000
        if not weap_built:
            tech_reserve += self._price("weap", context) or 2000
        # If one core building is complete but the next is already partially
        # constructing, the model may spend only money in excess of the full
        # conservative reserve. This errs toward healthy production capacity.
        accepted: list[dict] = []
        used_units: set[int] = set()
        infrastructure_commands = 0
        for order in valid:
            item = str(order.get("item") or "").lower()
            cmd = order["cmd"]
            key = self._order_key(order)
            if (self.clock() < self.sent_recent.get(key, -1e9) or
                    self.clock() < self.refused_recent.get(key, -1e9)):
                bad += 1
                continue
            debit = 0
            if cmd in ("build", "train"):
                if item not in available:
                    bad += 1
                    continue
                if cmd == "build":
                    if infrastructure_commands >= 1 or queued[item] > 0:
                        bad += 1
                        continue
                if cmd == "train" and queued[item] >= 5:
                    bad += 1
                    continue
                count = order.get("count", 1) if cmd == "train" else 1
                price = self._price(item, context)
                if price is None:
                    # Let server determine unknown build prices, but do not
                    # permit unknown-priced unit batches to drain key reserves.
                    if cmd == "train" and tech_reserve and not under_attack:
                        bad += 1
                        continue
                else:
                    cost = price * count
                    if cmd == "train" and budget + 1e-6 < cost:
                        bad += 1
                        continue
                    # A building may be accepted into the construction queue
                    # before its full price has been paid (engine-dependent).
                    # Do not impose a stricter client-side price rule: send the
                    # build and learn the server's actual acceptance/refusal.
                    if cmd == "train" and not under_attack and item != "harv" and tech_reserve:
                        # One key building has not yet been funded. Avoid the
                        # original economic death spiral from queued infantry.
                        if budget - cost < tech_reserve:
                            bad += 1
                            continue
                    debit = min(budget, cost)
            elif cmd == "place":
                if item not in ready:
                    bad += 1
                    continue
            elif cmd == "deploy":
                ids = order.get("units", [])
                if any(i not in units or kind(units[i]) != "mcv" for i in ids):
                    bad += 1
                    continue
            if "units" in order:
                ids = order["units"]
                if any(i not in units for i in ids):
                    bad += 1
                    continue
                # Two orders for the same unit in one batch cancel each other.
                if used_units.intersection(ids):
                    bad += 1
                    continue
            if cmd in ("repair", "power_toggle", "sell", "set_primary", "rally"):
                if order.get("building") not in own_buildings:
                    bad += 1
                    continue
            if cmd == "attack" and "target" in order and order["target"] not in enemies:
                bad += 1
                continue
            if cmd in ("guard", "enter_transport") and order.get("target") not in units and order.get("target") not in own_buildings:
                bad += 1
                continue
            if bounds is not None:
                # Bounds are [x, y, width, height] (may be offset).
                if isinstance(bounds, list) and len(bounds) == 4 and all(integer(x) for x in bounds):
                    outside = any(isinstance(order.get(field), list) and
                        len(order[field]) == 2 and not (
                            bounds[0] <= order[field][0] < bounds[0] + bounds[2] and
                            bounds[1] <= order[field][1] < bounds[1] + bounds[3])
                        for field in ("to", "at"))
                    if outside:
                        bad += 1
                        continue
            accepted.append(order)
            budget -= debit
            if cmd == "build":
                infrastructure_commands += 1
                queued[item] += 1
            elif cmd == "train":
                queued[item] += order.get("count", 1)
            if "units" in order:
                used_units.update(order["units"])
        return accepted, bad

    def _send(self, orders: list[dict], reason: str, *, cooldown=4.0):
        if not orders:
            return WAIT
        now = self.clock()
        for o in orders:
            self.sent_recent[self._order_key(o)] = now + (12 if o.get("cmd") in ("build", "train") else cooldown)
        self.last_batch = list(orders)
        summary = reason[:280]
        self.log("[riffo7] " + "; ".join(f"{o.get('cmd')} {o.get('item','')} {o.get('count','')}".strip() for o in orders[:10]) + " | " + summary)
        return WithReasoning({"type": "orders", "orders": orders}, summary)

    def _quick(self, raw: dict, obs: dict) -> Any:
        now = self.clock()
        # These actions do not require strategy and are costly to postpone.
        for u in rows(obs.get("units")):
            if kind(u) == "mcv" and integer(u.get("id")):
                order = {"cmd": "deploy", "units": [u["id"]]}
                if now >= self.sent_recent.get(self._order_key(order), -1e9):
                    return self._send([order], "Deploy MCV immediately")
        for p in rows(obs.get("production")):
            if p.get("ready") is True and str(p.get("item", "")).lower() in self.BUILDINGS:
                order = {"cmd": "place", "item": str(p["item"]).lower()}
                if now >= self.sent_recent.get(self._order_key(order), -1e9):
                    return self._send([order], "Place finished structure immediately")
        # A harvester marked idle should always be harvesting.
        for u in rows(obs.get("units")):
            if kind(u) not in ("harv", "harvester", "miner") or not integer(u.get("id")):
                continue
            stalled = now - self.last_cash_increase >= 35
            # In a long zero-income deadlock, try an explicit harvest command
            # occasionally even when 'idle' was falsely reported as False.
            if u.get("idle") is True or stalled:
                id_ = u["id"]
                if now >= self.last_harvest_rescue.get(id_, -1e9) + (35 if stalled else 14):
                    self.last_harvest_rescue[id_] = now
                    return self._send([{"cmd": "harvest", "units": [id_]}],
                                      "Restart inactive or stalled harvesting")
        return WAIT

    def _consume(self, raw: dict, context: Any):
        now = self.clock()
        while True:
            try:
                gen, started, result, error = self.replies.get_nowait()
            except queue.Empty:
                return WAIT
            if gen != self.generation:
                continue
            self.worker_active = False
            if error:
                self.errors += 1
                pause = min(40, 5 * (2 ** min(self.errors - 1, 3)))
                self.next_call = max(self.next_call, now + pause)
                self.log(f"[riffo7] AI error: {error}; retry in {pause}s (automation remains active)")
                continue
            self.errors = 0
            # A tactical plan from an old game state is dangerous in real time.
            if now - started > max(12.0, self.timeout + 2):
                self.log("[riffo7] discarded outdated model decision")
                continue
            offered = result.get("orders", [])
            if not isinstance(offered, list):
                self.log("[riffo7] AI reply omitted order list")
                continue
            orders, dropped = self._safe_orders(offered, raw, context)
            self.last_reason = str(result.get("reasoning_summary") or "AI directed orders")[:280]
            if dropped:
                self.log(f"[riffo7] AI proposed {len(offered)} orders; skipped {dropped} stale, unsafe or malformed")
            if orders:
                self.empty_answers = 0
                return self._send(orders, self.last_reason)
            self.empty_answers += 1
            if self.empty_answers <= 2 or self.empty_answers % 5 == 0:
                self.log(f"[riffo7] AI WAIT ({len(offered)} proposed, {dropped} skipped): {self.last_reason}")
            if offered:
                self.next_call = min(self.next_call, now + 1)
            return WAIT

    def _outage_fallback(self, raw: dict, obs: dict, context: Any):
        """Very limited recovery ONLY after repeated LLM failures.

        Never replaces an available model's strategy. Keep core construction
        moving if the API is unavailable so a temporary 429 isn't a forfeit.
        """
        if self.errors < 2:
            return WAIT
        names = {kind(b) for b in rows(obs.get("buildings"))}
        in_queue = {str(p.get("item") or "").lower() for p in rows(obs.get("production"))}
        offered = {i.lower() for i in obs.get("available_production") or [] if isinstance(i, str)}
        econ = obj(obs.get("economy"))
        power, draw = econ.get("power_provided"), econ.get("power_drained")
        if isinstance(power, (float, int)) and isinstance(draw, (float, int)) and power > 0 and draw >= power:
            preferred = ("powr", "apwr")
        elif not any(b in names or b in in_queue for b in ("powr", "apwr")):
            preferred = ("powr", "apwr")
        elif not any(b in names or b in in_queue for b in ("barr", "tent")):
            preferred = ("barr", "tent")
        elif "proc" not in names and "proc" not in in_queue:
            preferred = ("proc",)
        elif "weap" not in names and "weap" not in in_queue:
            preferred = ("weap",)
        else:
            return WAIT
        for item in preferred:
            if item in offered and item not in in_queue:
                action = {"cmd": "build", "item": item}
                valid, _ = self._safe_orders([action], raw, context)
                if valid:
                    return self._send(valid, "Provider outage: maintain essential infrastructure")
        return WAIT

    def choose_action(self, state: Any, context: Any):
        raw = obj(getattr(state, "raw", {}))
        obs = obj(raw.get("observation"))
        if not obs:
            return WAIT
        self._observe(raw, obs)
        self._feedback(raw)
        urgent = self._quick(raw, obs)
        # Start or reap AI calls even when a mandatory action was sent.
        if urgent is not WAIT:
            self._request(raw, obs, context)
            return urgent
        model_action = self._consume(raw, context)
        if model_action is WAIT:
            model_action = self._outage_fallback(raw, obs, context)
        self._request(raw, obs, context)
        now = self.clock()
        if now - self.last_status >= 12:
            self.last_status = now
            e = obj(obs.get("economy"))
            ours = rows(obs.get("units"))
            enemy = rows(obs.get("visible_enemies"))
            harvesters = [{"id":u.get("id"), "idle":u.get("idle"), "cell":_cell(u),
                           "order":u.get("order", u.get("activity"))} for u in ours
                          if kind(u) in ("harv", "harvester", "miner")]
            self.log("[riffo7] STATUS cash=%s power=%s/%s buildings=%s army=%s visible_enemy=%s "
                     "harvesters=%s ai_calls=%s errors=%s" % (
                         e.get("cash"), e.get("power_provided"), e.get("power_drained"),
                         dict(Counter(kind(b) for b in rows(obs.get("buildings")))),
                         sum(kind(u) not in self.NONCOMBAT for u in ours), len(enemy),
                         short(harvesters, 400), self.ai_calls, self.errors))
        return model_action

    def on_action_result(self, result: dict, context: Any) -> None:
        if isinstance(result, dict) and (result.get("error") == "INVALID_ACTION" or result.get("accepted") is False):
            self.log("[riffo7] batch rejected: " + short({k: result.get(k) for k in ("error", "detail", "message")}, 350))
            for order in self.last_batch:
                self.refused_recent[self._order_key(order)] = self.clock() + 12
            self.last_results.append("Server rejected entire batch: " + short(result, 280))
        self.last_batch = []
