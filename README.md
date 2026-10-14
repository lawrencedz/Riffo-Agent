# Riffo Agent — UCLA AltruAgent Tournament

Riffo is a self-hosted, three-game AI agent built on the official UCLA AltruAgent starter kit. It plays **Pokémon VGC doubles draft**, **Werewolf**, and **Red Alert** using a hybrid of structured LLM decisions and deterministic strategies.

> **Status:** Experimental v2. Local strategy tests can pass without demonstrating game wins. Validate at least one live Testing match before relying on it competitively.

## Strategy

| Game | Agent approach | Why |
|---|---|---|
| Pokémon VGC doubles | Groq LLM with game-specific draft, team-preview, and battle advice | Uses UCLA's legal-action adapters and Showdown timer fallbacks |
| Werewolf | Role-aware Groq LLM | Accounts for investigations, voting, hidden roles, and team objectives |
| Red Alert | Immediate deterministic build/attack rules + up to 12 short Groq calls per match | Handles real-time constraints and reduces API usage |

The contest server remains the authority on valid moves. This project intentionally relies on the starter's `examples/llm_agent.py`, `examples/llm/pokemon.py`, and `examples/llm/redalert.py` for structured action validation and game integration.

## Set up (Windows PowerShell)

Run commands in the folder containing `pyproject.toml`, which must also have `agent/`, `altruagent/`, and `examples/`.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Place these settings in **`.env`** (never commit or upload that file):

```dotenv
ALTRUAGENT_CONTROL_URL=https://api.altruagent-game.com
ALTRUAGENT_OFFICIAL_AGENT_KEY=eak_live_YOUR_PRIVATE_KEY
OPENAI_API_KEY=YOUR_PRIVATE_GROQ_KEY
OPENAI_MODEL=openai/gpt-oss-20b
OPENAI_BASE_URL=https://api.groq.com/openai/v1
```

`OPENAI_API_KEY` refers to Groq because the starter uses an OpenAI-compatible API interface.

## Check and play

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_riffo_agent.py
.\.venv\Scripts\python.exe -m agent --check-tournament
.\.venv\Scripts\python.exe -m agent --match
```

Create a game on the [Testing page](https://platform.altruagent-game.com/tournament/dashboard) with seats set to **Mine (self-hosted)**. The agent should say it has connected and is waiting for an assignment. That alone does not show that the Groq API works: inspect live decisions and the finished match.

To play both test and tournament games, launch **one process**:

```powershell
.\.venv\Scripts\python.exe -m agent --tournament --match
```

Register on the dashboard, keep the terminal and computer running during the scheduled tournament, and stop using `Ctrl+C` when appropriate.

## Safety and cost

- Rotate any keys that have appeared in screenshots or chats; save replacements only in local `.env`.
- `.env`, variants of `.env`, and `.venv` must be ignored by Git.
- A live Pokémon or Werewolf match consumes Groq calls. Red Alert normally uses rules and caps AI requests at 12 per match, up to eight seconds each.
- Groq free-tier rate limits and model response issues can still cause fallbacks. Watch console logs for `HTTP 400` / `429` and `[llm] ... FALLBACK`.

## Development

- `agent/agent.py`: Riffo's strategy orchestration.
- `tests/test_riffo_agent.py`: offline strategy and wiring tests.
- `agent/__main__.py`: official starter entry point (unchanged).
- `examples/` and `altruagent/`: the original tournament SDK and game adapters (unchanged).

## Evaluation priorities

1. Verify a self-hosted **Werewolf** match reaches completion.
2. Verify Pokémon **draft → team preview → doubles battle** completes without errors or timer forfeits.
3. Inspect Red Alert logs and refine build orders after actual observations.
4. Record win rates, decision times, invalid actions, and Groq API errors. None of these are validated by offline tests alone.
