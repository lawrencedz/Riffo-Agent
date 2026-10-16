# Riffo 6.3 — cumulative pre-tournament package

Includes v6.1 model-led Pokémon and evidence-driven Werewolf, **v6.2 Red Alert economic throughput fix**, plus small v6.3 Pokémon battle-mechanics checks. You need only **this ZIP**; do not install earlier versions first.

## Why this change

The latest battle against Abu Bakar Siddik revealed two strong, game-mechanics errors:

- **Turn 4:** Rillaboom selected **Fake Out** against Hatterene after remaining on the field since Turn 1. Fake Out fails when the user did not just enter the field.
- **Turns 6 and 7:** Iron Bundle selected **Protect** on two consecutive turns. The second consecutive Protect attempt is much less likely to succeed (approximately 1/3 in standard Gen 9 mechanics), so the bot must explicitly weigh that risk rather than treating it as reliable protection.

The opening double-Fake-Out decision is also risky when both leads can be heavily damaged if Fake Out does not cause a flinch; we do **not** know from the displayed action history whether the first turn's Fake Outs failed, which items/abilities were active, or exactly how damage resolved.

## Technical changes

- Tracks which *observed active species* occupied slot 0 and slot 1 across consecutive battle decisions. Unknown, fainted, or replaced slots do not inherit a previous Pokémon's state.
- Adds model-visible, factual warnings for a **non-fresh Fake Out** or **back-to-back Protect/Detect**. All server legal options remain visible; no static species rankings or arbitrary score tables have been added.
- When an LLM still chooses one of these risky moves, requests **one additional model correction** (up to 4.5 seconds) *only if* the battle-decision budget permits. The server-valid original is retained if correction times out or fails. This deliberately does not guarantee improved tactical performance when the LLM API is unavailable.
- Explicitly encourages comparing opponent Trick Room and Tailwind plans at preview, and treats double Fake Out as a contingent tactic rather than guaranteed protection.
- **Werewolf remains unchanged**: fully model-led role-sensitive evidence discussion, reliable wolf-teammate protection, and official legal-action validation.
- **Red Alert remains v6.2**: nonblocking high-level model strategy, economic reinvestment, timely order batches, per-order diagnostics.

## PowerShell install (Windows)

Stop the existing agent (Ctrl+C) and **download the ZIP**. The commands assume Downloads is your browser's download location:

```powershell
cd "C:\Users\larry\OneDrive\Documents\Riffo Agent\Riffo-Agent"
$zip = "$HOME\Downloads\Riffo-Agent-v6_3-complete-final.zip"
$patch = "$env:TEMP\riffo-v63"
if (!(Test-Path $zip)) { throw "Download the v6.3 ZIP to Downloads first." }
Expand-Archive -LiteralPath $zip -DestinationPath $patch -Force
Copy-Item .\agent\agent.py "$env:TEMP\riffo-before-v63.py" -Force
Copy-Item "$patch\agent\agent.py" .\agent\agent.py -Force
foreach ($n in @('test_riffo_v60.py','test_riffo_v61.py','test_riffo_v62.py','test_riffo_v63.py')) {
    Copy-Item "$patch\tests\$n" ".\tests\$n" -Force
}
Copy-Item "$patch\scripts\setup_pokemon_reference.py" .\scripts\setup_pokemon_reference.py -Force
.\.venv\Scripts\python.exe -m py_compile .\agent\agent.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_riffo_v60.py tests/test_riffo_v61.py tests/test_riffo_v62.py tests/test_riffo_v63.py
.\.venv\Scripts\python.exe -m agent --check-tournament
```

The four focused test files passed **30/30 in an offline stubbed SDK harness**, not in a live UCLA match. Expect `30 passed` under compatible dependencies. The `--check-tournament` step needs live platform access and a valid tournament key.

### Optional reference setup

Once per hosting machine (only if the reference JSON isn't already present):

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_pokemon_reference.py
```

This downloads public Pokémon Showdown move/species data and saves `agent/pokemon_reference.json` locally. This reference is not included in this ZIP. Do not assume the model has verified typings/move powers until the script succeeds. Copy the JSON to your continuously online hosting computer if that computer cannot download it.

### Roll back

```powershell
Copy-Item "$env:TEMP\riffo-before-v63.py" .\agent\agent.py -Force
```

### Commit safely

Confirm `.env` is **not tracked**, and no secrets have ever entered Git history before pushing:

```powershell
git ls-files .env
git log --all --format= --name-only -- .env
# If either command outputs .env, STOP and address the exposed credentials.
git add agent/agent.py tests/test_riffo_v60.py tests/test_riffo_v61.py tests/test_riffo_v62.py tests/test_riffo_v63.py scripts/setup_pokemon_reference.py
git diff --cached --name-only
git commit -m "Riffo v6.3 AI-first cumulative tournament update"
git push -u origin main
```

A GitHub push **does not start the runner**. Self-hosted tournament play requires a computer that stays online, registered on the platform, running **one** copy of:

```powershell
.\.venv\Scripts\python.exe -m agent --tournament
```

On a Linux cloud VM use `./.venv/bin/python -m agent --tournament` instead. If your laptop will disconnect tomorrow, install and start this version on the always-online machine before you leave.

## Limitations and priorities

- No v6.3 live win-rate evidence. The new Pokemon tracking is best-effort and depends on the real template exposing the active species. If not present, it will not guess.
- No changes to the server, game rules, or user's .env/API keys.
- The optional correction adds a model call only when it sees a severe tactical choice and time remains; if that fails, the previously validated legal choice is returned.
- Most urgent: verify test passes, actual platform readiness, internet-connected hosting, and what code version is loaded before the tournament.
