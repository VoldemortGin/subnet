# HARM — Hey Asshole, Return Money

<p align="center">
  <img src="docs/harm_icon_main.png" alt="HARM Icon" width="256">
</p>

Decentralized image forgery detection for e-commerce refund fraud. A Bittensor subnet proposal.

![HARM Screenshot](docs/ScreenShot_2026-05-23_170640_167.png)

## Quick Start

```bash
# 1. Install backend dependencies
pip install -r backend/requirements.txt

# 2. Start backend (port 8000)
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000

# 3. Start frontend (port 5173)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

## Usage

- **Dashboard** — Network stats, recent analyses, miner leaderboard
- **Analyze** — Upload a suspicious image, get verdict (authentic/tampered) + confidence + detection overlay
- **Submit** — Simulate the full Miner Commit-Reveal workflow: accept task → analyze → compute SHA-256 hash → commit → reveal → get scored
- **Miners** — View registered miners, probe accuracy, strike status
- **Tasks** — Browse probe and real tasks, generate new probes

## Project Structure

```
backend/     — FastAPI server (image analysis, scoring, miner management)
frontend/    — React + Tailwind UI
src/         — Core subnet logic (forge engine, scorer, detection backends)
proposal/    — Pitch deck and flowchart presentations
docs/        — Proposal document and architecture plans
```
