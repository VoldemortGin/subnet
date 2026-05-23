# HARM Full-Stack Implementation Plan

## 1. Complete File List

### Backend (`backend/`)

```
backend/
├── main.py                    # FastAPI app, CORS, static mounts, lifespan
├── api/
│   ├── __init__.py
│   ├── deps.py                # Shared dependencies: get_store(), get_services()
│   ├── models.py              # Pydantic request/response schemas
│   └── routes/
│       ├── __init__.py
│       ├── images.py          # POST /upload, GET /{id}, GET /{id}/visualization
│       ├── miners.py          # POST /register, GET /, POST /{id}/submit
│       ├── tasks.py           # GET /, POST /probe
│       └── dashboard.py       # GET /stats, GET /leaderboard
├── services/
│   ├── __init__.py
│   ├── task_service.py        # Create tasks, distribute to miners, lifecycle
│   ├── miner_service.py       # Registration, scoring, strike system
│   ├── forge_service.py       # Wraps ForgeEngine for probe generation
│   └── detect_service.py      # Wraps ForgeryDetector for built-in analysis
├── db/
│   ├── __init__.py
│   └── store.py               # In-memory dict store (all state)
└── requirements.txt           # Backend-only deps
```

### Frontend (`frontend/`)

```
frontend/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.ts
├── postcss.config.js
├── index.html
└── src/
    ├── main.tsx               # ReactDOM entry
    ├── App.tsx                # Router setup, layout wrapper
    ├── api/
    │   └── client.ts          # Typed fetch wrapper for all endpoints
    ├── components/
    │   ├── Layout.tsx          # Sidebar + top bar + content area
    │   ├── ImageUpload.tsx     # Drag-and-drop upload zone
    │   ├── ResultCard.tsx      # Single analysis result (verdict badge, confidence bar, mask overlay)
    │   ├── MinerTable.tsx      # Sortable miner leaderboard table
    │   ├── StatsCards.tsx      # 4-card grid (total tasks, miners, avg score, fraud rate)
    │   ├── ScoreBreakdown.tsx  # Stacked bar showing probe/consensus/latency split
    │   ├── TaskRow.tsx         # Single task row for task list (probe badge, status, result)
    │   └── StrikeIndicator.tsx # Visual 3-strike display (filled/empty circles)
    ├── pages/
    │   ├── Dashboard.tsx       # Stats + recent analyses + mini leaderboard
    │   ├── Analyze.tsx         # Upload image -> show result with mask overlay
    │   ├── Miners.tsx          # Full miner table + registration form
    │   ├── Tasks.tsx           # Task list with filters (probe/real, status)
    │   └── Demo.tsx            # Step-by-step walkthrough of full HARM flow
    ├── hooks/
    │   └── useApi.ts           # Generic fetch hook with loading/error state
    └── styles/
        └── globals.css         # Tailwind directives + HARM theme tokens
```

---

## 2. Dependencies

### Python (backend/requirements.txt)

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9         # file uploads
pydantic>=2.6.0
```

Plus existing project deps already in pyproject.toml:
- opencv-python, Pillow, numpy, scikit-image

### npm (frontend/package.json)

```json
{
  "dependencies": {
    "react": "^18.3",
    "react-dom": "^18.3",
    "react-router-dom": "^6.23"
  },
  "devDependencies": {
    "@types/react": "^18.3",
    "@types/react-dom": "^18.3",
    "@vitejs/plugin-react": "^4.3",
    "autoprefixer": "^10.4",
    "postcss": "^8.4",
    "tailwindcss": "^3.4",
    "typescript": "^5.5",
    "vite": "^5.4"
  }
}
```

---

## 3. File Descriptions

### Backend

| File | Contents |
|---|---|
| `main.py` | FastAPI app factory. Mounts `/api` routes. Adds CORS middleware (allow `localhost:5173`). Mounts `/uploads` as static directory. Lifespan hook seeds 3 demo miners + 5 probe tasks on startup. |
| `api/models.py` | Pydantic models: `ImageUploadResponse`, `AnalysisResult` (verdict, confidence, method, mask_url, visualization_url), `MinerInfo` (id, name, backend, score, strikes, active), `TaskInfo` (id, type, status, result), `DashboardStats`, `LeaderboardEntry`, `MinerRegisterRequest` (name, backend_name), `MinerSubmitRequest` (verdict, confidence, method). |
| `api/deps.py` | `get_store()` returns singleton `InMemoryStore`. `get_forge_service()`, `get_detect_service()`, `get_task_service()`, `get_miner_service()` factory functions. |
| `api/routes/images.py` | `POST /api/images/upload` -- accepts multipart file, saves to `uploads/`, runs detection, returns `AnalysisResult`. `GET /api/images/{id}` -- returns stored result. `GET /api/images/{id}/visualization` -- generates overlay image (original + red mask) and returns as PNG response. |
| `api/routes/miners.py` | `POST /api/miners/register` -- creates miner entry with chosen backend. `GET /api/miners` -- lists all miners with stats. `POST /api/miners/{id}/submit` -- miner submits detection result for an assigned task; triggers scoring. |
| `api/routes/tasks.py` | `GET /api/tasks` -- lists all tasks with optional `?type=probe&status=pending` filters. `POST /api/tasks/probe` -- uses ForgeEngine to generate a new probe task from a random clean image. |
| `api/routes/dashboard.py` | `GET /api/dashboard/stats` -- returns totals (tasks, miners, avg score, fraud detection rate). `GET /api/dashboard/leaderboard` -- returns miners sorted by total_score desc, top 20. |
| `services/task_service.py` | `create_probe_task()` -- calls ForgeEngine, stores task + ground truth. `create_real_task(image_path)` -- creates task from user upload. `assign_task(task_id, miner_id)` -- marks task assigned. `get_task(id)`, `list_tasks(filters)`. |
| `services/miner_service.py` | `register_miner(name, backend)` -- creates miner record. `record_submission(miner_id, task_id, response)` -- stores result, if probe: scores via ProbeScorer and updates strike window. `compute_strikes(miner_id)` -- checks last 10 probes, strike if score < 0.3 on any, deactivate if 3+ strikes. `get_leaderboard()`. |
| `services/forge_service.py` | Thin wrapper around `src.validator.forge.ForgeEngine`. `generate_probe(clean_dir)` -- picks random clean image, calls `engine.generate_probe()`, returns paths + ground truth. |
| `services/detect_service.py` | Thin wrapper around `src.miner.detector.ForgeryDetector`. `analyze_image(image_path)` -- runs detection with ELA backend (always available), returns `MinerResponse`. `generate_visualization(image_path, mask)` -- creates red-overlay composite image. |
| `db/store.py` | `InMemoryStore` class with dicts: `tasks: dict[str, TaskRecord]`, `miners: dict[str, MinerRecord]`, `images: dict[str, ImageRecord]`, `submissions: dict[str, list[SubmissionRecord]]`. Simple dataclass records. No ORM. |

### Frontend

| File | Contents |
|---|---|
| `App.tsx` | `BrowserRouter` with routes: `/` -> Dashboard, `/analyze` -> Analyze, `/miners` -> Miners, `/tasks` -> Tasks, `/demo` -> Demo. Wraps in `<Layout>`. |
| `api/client.ts` | `const API_BASE = "http://localhost:8000/api"`. Typed functions: `uploadImage(file)`, `getImage(id)`, `getVisualization(id)`, `listTasks(filters?)`, `createProbe()`, `registerMiner(name, backend)`, `listMiners()`, `submitResult(minerId, taskId, data)`, `getStats()`, `getLeaderboard()`. Returns typed responses. |
| `components/Layout.tsx` | Dark sidebar (HARM logo, nav links with icons), top bar (project name), main content area. Fixed sidebar, scrollable content. |
| `components/ImageUpload.tsx` | Drag-and-drop zone with file input fallback. Shows preview thumbnail. Calls `uploadImage()` on drop. Displays loading spinner during upload. |
| `components/ResultCard.tsx` | Takes `AnalysisResult`. Shows verdict badge (green AUTHENTIC / red TAMPERED), confidence progress bar, detected method chip, and mask overlay toggle (loads visualization URL). |
| `components/MinerTable.tsx` | Takes `MinerInfo[]`. Columns: rank, name, backend, probe score, consensus score, total score, strikes (StrikeIndicator), status badge. Sortable by score columns. |
| `components/StatsCards.tsx` | 4-card grid. Each card: icon, label, value, trend indicator. Cards: Total Tasks, Active Miners, Avg Score, Fraud Rate. |
| `components/ScoreBreakdown.tsx` | Takes a single miner's scores. Horizontal stacked bar: probe (60%), consensus (35%), latency (5%). Labeled segments with values. |
| `components/TaskRow.tsx` | Single row: task ID (truncated), type badge (PROBE/REAL), status (pending/complete), verdict if complete, timestamp. Click to expand details. |
| `components/StrikeIndicator.tsx` | Takes `strikes: number, max: number`. Renders filled red circles for strikes, gray outlines for remaining. |
| `pages/Dashboard.tsx` | Fetches `/stats` and `/leaderboard` on mount. Renders StatsCards at top, mini MinerTable (top 5), recent task list (last 10). Auto-refreshes every 10s. |
| `pages/Analyze.tsx` | ImageUpload component at top. Below: result area (empty state -> ResultCard after upload). Side panel: detection details (ELA heatmap, noise map if available). |
| `pages/Miners.tsx` | Registration form (name + backend dropdown from available backends). Full MinerTable below. Click row to expand ScoreBreakdown + submission history. |
| `pages/Tasks.tsx` | Filter bar (type: all/probe/real, status: all/pending/complete). "Generate Probe" button. Task list with TaskRow components. Pagination (20 per page). |
| `pages/Demo.tsx` | Step-by-step wizard: (1) Show clean image, (2) Generate forgery (calls probe endpoint), (3) Show tampered vs original, (4) Run detection, (5) Show scoring breakdown. Each step has a "Next" button. Animated transitions. |
| `hooks/useApi.ts` | Generic hook: `useApi<T>(fetcher) -> { data, loading, error, refetch }`. Handles AbortController cleanup. |
| `styles/globals.css` | Tailwind `@tailwind` directives. CSS variables: `--harm-red: #DC2626`, `--harm-bg: #0F0F0F`, `--harm-surface: #1A1A1A`, `--harm-border: #2A2A2A`, `--harm-text: #E5E5E5`. |

---

## 4. Integration Points

### API Contract (frontend <-> backend)

All communication via JSON over HTTP. Frontend proxied in dev via Vite config (`/api` -> `http://localhost:8000`).

| Frontend Action | Backend Endpoint | Data Flow |
|---|---|---|
| Upload image for analysis | `POST /api/images/upload` (multipart) | Returns `AnalysisResult` with verdict, confidence, mask URL |
| View mask overlay | `GET /api/images/{id}/visualization` | Returns PNG image (streamed) |
| Register miner | `POST /api/miners/register` | Sends `{name, backend_name}`, returns `MinerInfo` |
| View leaderboard | `GET /api/dashboard/leaderboard` | Returns `MinerInfo[]` sorted by score |
| Generate probe | `POST /api/tasks/probe` | Backend calls ForgeEngine, creates task, returns `TaskInfo` |
| Miner submits result | `POST /api/miners/{id}/submit` | Sends verdict+confidence, backend scores via ProbeScorer |
| Dashboard refresh | `GET /api/dashboard/stats` | Returns aggregated counts and averages |

### Image Serving

- Uploaded images saved to `backend/uploads/` directory
- Served via FastAPI `StaticFiles` mount at `/uploads`
- Visualization images (mask overlays) generated on-demand and cached in `backend/uploads/viz/`
- Frontend references images via `/uploads/{filename}` URLs

### Masks

- Masks stored as numpy arrays in memory (backend store)
- Sent to frontend as PNG images via the visualization endpoint (red overlay on original)
- Never sent as raw numpy -- always rendered server-side

---

## 5. Existing `src/` Reuse Map

| Existing File | Used By | How |
|---|---|---|
| `src/protocol.py` | `api/models.py`, all services | `Verdict`, `ForgeryMethod` enums used directly in API models. `MinerResponse`, `GroundTruth`, `ProbeTask`, `ScoreResult` dataclasses used internally in services. |
| `src/validator/forge.py` | `services/forge_service.py` | `ForgeEngine.generate_probe()` called to create probe tasks. `ForgeEngine.forge()` called for the demo page's live forgery generation. |
| `src/validator/scorer.py` | `services/miner_service.py` | `ProbeScorer.score_probe()` called when a miner submits a result for a probe task. `ProbeScorer.score_epoch()` called to compute epoch-level scores for the leaderboard. `ScoringWeights` used for displaying weight breakdown in the UI. |
| `src/validator/generate_samples.py` | `main.py` (startup) | `generate_clean_images()` called during lifespan startup to ensure clean images exist in `data/clean/`. Not called at runtime. |
| `src/miner/detector.py` | `services/detect_service.py` | `ForgeryDetector.detect()` and `detect_from_path()` called for the built-in analysis when a user uploads an image. The backend itself acts as one "miner" using this. |
| `src/miner/model_registry.py` | `api/routes/miners.py`, `services/miner_service.py` | `list_available()` provides the backend dropdown options for miner registration. `get_backend(name)` instantiates the chosen backend for a miner. |
| `src/miner/backends/base.py` | `services/detect_service.py` | `DetectionBackend` interface used for type hints. |
| `src/miner/backends/ela.py` | `services/detect_service.py` | `ELABackend` used as the default always-available backend for the built-in analysis endpoint. |

---

## 6. Build Order

### Phase 1: Backend skeleton (get API serving)

1. **`backend/db/store.py`** -- Define `InMemoryStore` with all record dataclasses. This is the foundation everything writes to.
2. **`backend/api/models.py`** -- Define all Pydantic schemas. Import and reuse `Verdict`/`ForgeryMethod` from `src.protocol`.
3. **`backend/services/forge_service.py`** -- Wrap `ForgeEngine`. Verify it can generate probes using existing clean images.
4. **`backend/services/detect_service.py`** -- Wrap `ForgeryDetector` with ELA backend. Verify it can analyze an image and return results.
5. **`backend/services/task_service.py`** -- Task CRUD + probe generation orchestration.
6. **`backend/services/miner_service.py`** -- Miner registration, submission handling, scoring, strike system.
7. **`backend/api/deps.py`** -- Wire up dependency injection.
8. **`backend/api/routes/`** -- All four route files. Mount in `main.py`.
9. **`backend/main.py`** -- App factory, CORS, static mounts, startup seeding.

**Milestone: `uvicorn backend.main:app` serves all endpoints. Test with curl/httpie.**

### Phase 2: Frontend skeleton (pages render)

10. **Scaffold** -- `npm create vite@latest frontend -- --template react-ts`. Install tailwind, react-router-dom.
11. **`styles/globals.css`** -- HARM dark theme.
12. **`api/client.ts`** -- All API functions with types.
13. **`hooks/useApi.ts`** -- Generic fetch hook.
14. **`components/Layout.tsx`** -- Sidebar navigation shell.
15. **`pages/Dashboard.tsx`** + **`components/StatsCards.tsx`** -- First visible page.

**Milestone: Frontend renders dashboard with live data from backend.**

### Phase 3: Core features

16. **`pages/Analyze.tsx`** + **`components/ImageUpload.tsx`** + **`components/ResultCard.tsx`** -- Image upload and analysis flow.
17. **`pages/Miners.tsx`** + **`components/MinerTable.tsx`** + **`components/StrikeIndicator.tsx`** + **`components/ScoreBreakdown.tsx`** -- Miner management.
18. **`pages/Tasks.tsx`** + **`components/TaskRow.tsx`** -- Task list and probe generation.

**Milestone: All CRUD operations work end-to-end.**

### Phase 4: Demo page + polish

19. **`pages/Demo.tsx`** -- Step-by-step walkthrough that ties everything together.
20. **Visualization endpoint** -- Mask overlay rendering, cached PNGs.
21. **Polish** -- Loading states, error boundaries, empty states, responsive breakpoints.

**Milestone: Demo-ready full-stack application.**

---

## Notes

- Backend runs on port 8000, frontend on port 5173 (Vite dev server).
- Vite config proxies `/api` to backend to avoid CORS in dev.
- No authentication needed -- this is a demo/prototype.
- All state is in-memory; restarting the backend resets everything (acceptable for demo).
- The `data/clean/` images (generated by `generate_samples.py`) are used as source material for probe generation.
- Only ELA backend is guaranteed available (no GPU deps). Other backends shown as "unavailable" in the UI unless their deps are installed.
