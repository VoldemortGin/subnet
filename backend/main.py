from __future__ import annotations

import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import dashboard, images, miners, tasks
from backend.db.store import get_store
from backend.services.forge_service import ForgeService
from backend.services.miner_service import MinerService
from backend.services.task_service import TaskService
from backend.services.detect_service import DetectService

DATA_DIR = Path("data")


def _seed_demo_data():
    store = get_store()
    miner_svc = MinerService(store)
    forge_svc = ForgeService(store, DATA_DIR)
    detect_svc = DetectService(store, DATA_DIR)
    task_svc = TaskService(store, forge_svc, detect_svc, miner_svc)

    demo_miners = [
        ("AlphaDetector", "ela"),
        ("ManTraNet-Node", "mantranet"),
        ("TruFor-Sentinel", "trufor"),
        ("Alice (Human)", "human_alice"),
        ("Bob (Human)", "human_bob"),
    ]
    miner_ids = []
    for name, backend in demo_miners:
        m = miner_svc.register_miner(name, backend)
        miner_ids.append(m.id)

    probe_histories: list[list[bool]] = [
        [True, True, True, True, True, True, True, True, False, True],
        [True, True, False, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True],
        [True, False, False, True, True, False, True, True, True, True],
        [True, True, True, True, True, True, True, False, True, True],
    ]

    for miner_id, history in zip(miner_ids, probe_histories):
        for correct in history:
            score = random.uniform(0.6, 1.0) if correct else random.uniform(0.0, 0.2)
            miner_svc.record_probe_result(miner_id, correct, score)

    clean_images = forge_svc.list_clean_images()
    if clean_images:
        for img_path in clean_images[:3]:
            try:
                task_svc.create_probe_task()
            except Exception:
                pass

    print(f"Seeded {len(miner_ids)} miners and {len(store.tasks)} probe tasks")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _seed_demo_data()
    yield


app = FastAPI(
    title="HARM - Decentralized Image Forgery Detection",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

app.include_router(images.router)
app.include_router(miners.router)
app.include_router(tasks.router)
app.include_router(dashboard.router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "project": "HARM"}
