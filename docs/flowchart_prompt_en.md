Generate a self-contained HTML file as a flowchart presentation. Requirements: single file, all CSS/JS inline, dark theme (black background + red #DC2626 accent + white text), left/right arrow keys to navigate, F key for fullscreen. One flowchart per page, 10 pages total.

Project name: HARM (Hey Asshole, Return Money) — a Bittensor subnet that uses a decentralized network to detect fake images in e-commerce refund fraud.

Core mechanism: Merchant submits suspicious refund image → Validator distributes to multiple Miners → Miners judge authenticity using ML models or human inspection → Validator scores results → On-chain emission distribution.

---

Page 1: System Architecture Overview

Participants:
- Bittensor Network (Yuma Consensus + Emission Distribution)
- Validator Nodes (Forge Engine + Task Distribution + Scorer)
- Miner Nodes (ML Model Miner / Human Miner / Hybrid Miner)
- Merchant Clients (submit suspicious images via API)

Flow:
Merchant → API → Validator → Distribute tasks to Miners → Miners return detection results → Validator scores → Submit weights to Bittensor chain → Emission distributed to Miners and Validators

---

Page 2: Validator Workflow

Steps:
1. Image source management: 70% real merchant queries + 30% probe tasks (self-generated fakes)
2. Forge Engine: generates fake images from clean originals using 4 tampering methods:
   - Copy-Move (duplicate a region within the same image)
   - Splicing (paste a region from another image)
   - Compression Mismatch (re-compress a region at different JPEG quality)
   - Noise Injection (add localized Gaussian noise)
   Each probe records: tampering method + ground truth mask
3. Task distribution: mix probes and real tasks, send to all Miners
4. Collect Miner responses
5. Scoring (probes scored against ground truth; real tasks scored by weighted vote)
6. Submit weights to Bittensor chain

---

Page 3: Miner Workflow

Steps:
1. Receive image from Validator
2. Choose detection method:
   - ML model path: choose from 10 model backends (Mesorch / TruFor / FOCAL / IML-ViT / ProFact / PSCC-Net, etc.)
   - Human path: visual inspection + manual annotation
   - Hybrid path: model pre-screening + human review of edge cases
3. Run detection analysis
4. Generate response: verdict (authentic/tampered) + confidence (0-1) + mask (tampered region, optional) + method (tampering type, optional)
5. Submit via Commit-Reveal protocol

---

Page 4: Commit-Reveal Anti-Copying Protocol

Timeline:
T0 — Validator distributes the same image to all Miners
T1 — Each Miner independently computes result, submits hash(result + nonce) (Phase 1: Commit)
T2 — All Miner hashes collected, commit window closes
T3 — Miners reveal original result + nonce (Phase 2: Reveal)
T4 — Validator verifies hash consistency → mismatches get zero score → scoring begins

Key point: After submitting their hash but before the reveal phase, Miners can only see irreversible hash values from others — they cannot reverse-engineer anyone's verdict, confidence, or mask. Copying is impossible.

---

Page 5: Scoring Pipeline

Four scoring dimensions:

A. Probe Task Score (weight: 60%) — ground truth available:
   Gate: verdict correct → continue scoring; verdict wrong → score = 0, and record one Strike
   After passing the gate:
   - Confidence Calibration (70%): C_cal = 1 - |confidence - ideal_value|
   - Tampered Region IoU (20%, optional bonus): intersection over union of predicted vs ground truth mask
   - Method Identification (10%, optional bonus): correctly identify the tampering method = 1 point

B. Consensus Task Score (weight: 35%) — no ground truth:
   Weighted majority vote, each Miner's vote weight = their historical probe accuracy
   Agree with weighted majority → 1 point; disagree → 0 points

C. Latency Score (weight: 5%):
   S_latency = max(0, 1 - response_time / timeout)
   Intentionally low weight to keep human Miners competitive

D. Strike System (Three Strikes, You're Out):
   Track each Miner's last 10 probe results (rolling window):
   - 0-2 wrong → Normal participation
   - 3 wrong → Yellow Card: consensus voting weight drops to 0, can only do probes to rebuild credit
   - 4 wrong → Red Card: weight set to 0, frozen for 3 epochs before recovery
   - 5+ wrong → Permanent Ban: kicked from network, must re-register (burn TAO)
   This is far more deterrent than simply reducing weights — speculators who guess wrong 3 times in 10 are immediately cut off.

Total Score = 0.60 x avg(Probe) + 0.35 x avg(Consensus) + 0.05 x Latency
(Prerequisite: Strike not triggered; otherwise total score = 0)

---

Page 6: Probe Task Generation Pipeline

Detailed steps:
1. Select a clean product image from the image library
2. Randomly choose a tampering method (Copy-Move / Splicing / Compression Mismatch / Noise Injection)
3. Apply the tampering operation to the image
4. Record ground truth: binary mask of tampered region + method used
5. Package the forged image and ground truth as a probe task
6. Mix into the task pool (Miners cannot distinguish probes from real tasks)
7. When network accuracy exceeds 90% for 3 consecutive epochs → automatically increase difficulty (smaller tampered regions, more advanced tampering methods)

---

Page 7: Anti-Gaming Overview

Display as a clean table or card layout, one attack per row. Left side: attack name and brief description. Right side: countermeasure name and brief description:

1. Sybil Attack (one person registers multiple Miners to manipulate voting) → Weighted voting + economic barrier
2. Free-Riding (wait for others to answer, then copy) → Commit-Reveal two-phase protocol
3. Always-Say-Fake (gamble on probability) → Dynamic real/fake ratio + gate + Strike System
4. Always-Say-Real (gamble on probability) → Same as above, fully symmetric
5. Model Overfitting (optimize only for Validator's forge patterns) → Method diversity + difficulty adaptation
6. Validator Collusion (leak probe answers to specific Miners) → Multi-Validator cross-verification + Yuma Consensus
7. Low-Quality Miners Squatting (persistent poor performance) → Strike System (Three Strikes, You're Out)

Keep this page concise — no more than two sentences per row.

---

Page 8: Anti-Gaming Detailed Explanations

Expand each defense mechanism with a dedicated card or block, bold title:

Defense 1: Weighted Voting
Consensus voting is weighted by each Miner's historical probe accuracy, not by identity count. An attacker controlling 5 Miners with low probe scores has less voting influence than a single high-scoring Miner. Bittensor registration requires burning ~2500 TAO, making large-scale Sybil attacks prohibitively expensive. 60% of the total score comes from probe tasks (objective evaluation), where Sybil identities gain no advantage.

Defense 2: Commit-Reveal Protocol
Phase 1 (Commit): All Miners submit hash(result + nonce) — an irreversible cryptographic commitment that cannot be modified after submission. Phase 2 (Reveal): Only after all hashes are collected does the reveal window open, where Miners disclose their original results and nonces. Validator verifies hash consistency — mismatches result in zero score, as does timeout. Between commit and reveal, anyone can only see irreversible hash values, making it impossible to reverse-engineer verdict or confidence.

Defense 3-4: Dynamic Real/Fake Ratio
The ratio of authentic to tampered images in probes is dynamically adjusted by the Validator each epoch (e.g., 70% real / 30% fake one round, reversed the next), and the ratio is never disclosed. An "always-say-fake" Miner triggers the gate (P_correct=0) on every authentic probe, dragging long-term accuracy below 50%. "Always-say-real" works identically. Validators can also deliberately increase the fake ratio to flush out speculators faster. Such Miners accumulate low historical probe scores, making their consensus voting weight approach zero.

Defense 5: Method Diversity + Difficulty Adaptation
The Forge Engine randomly selects from 4 tampering methods (Copy-Move / Splicing / Compression Mismatch / Noise Injection), randomly combined each round, making prediction impossible. When network accuracy exceeds 90% for 3 consecutive epochs, difficulty auto-escalates: smaller tampered regions, AI inpainting, multi-method combinations, post-processing. The 35% consensus weight rewards performance on real merchant images — Miners who overfit to probes but fail on real images are penalized on this dimension.

Defense 6: Multi-Validator Cross-Verification
The network has multiple independent Validators, each generating their own probes and scoring independently. If Validator A colludes with Miner X, X only scores high on A's probes but performs normally on B's and C's. Yuma Consensus detects A's anomalous weight distribution and automatically reduces A's influence. The colluding Validator's own emission is also reduced due to anomalous weights.

Defense 7: Strike System (Three Strikes, You're Out)
Track each Miner's last 10 probe verdicts (rolling window). 0-2 wrong: normal participation. 3 wrong: Yellow Card (consensus weight drops to 0, can only do probes to rebuild credit). 4 wrong: Red Card (weight zeroed, frozen for 3 epochs). 5+ wrong: Permanent Ban (kicked from network, must re-register by burning TAO). This ensures low-quality Miners are rapidly eliminated rather than slowly starved. Reducing weights is "death by a thousand cuts"; the Strike System is "immediate dismissal."

---

Page 9: Miner Resource Selection Guide

Decision tree:

Do you have a GPU?
├── No → Are you willing to manually inspect images?
│         ├── Yes → Human Miner (visual judgment + manual annotation)
│         └── No → PSCC-Net (2M params, runs on CPU, 0.019s per image)
│
├── Yes, 4-6 GB VRAM → Mesorch-P (AAAI 2025 latest SOTA, AUC 0.943)
│                        or TruFor (CVPR 2023, AUC 0.924)
│
├── Yes, 6-8 GB VRAM → IML-ViT (highest standardized F1: 0.725)
│
└── Yes, 8+ GB VRAM → FOCAL (strongest IoU)
                       or multi-model ensemble voting (accuracy ceiling)

---

Page 10: Business Loop

Complete business cycle:

1. Merchant receives refund request + suspicious image on e-commerce platform
2. Merchant submits image via HARM API
3. Validator mixes image into task pool (distributed alongside probes)
4. Multiple Miners independently analyze (ML + human)
5. Commit-Reveal collects results
6. Validator scores: probes against ground truth, real tasks by consensus
7. Forensic report returned to merchant: authentic/tampered + confidence + suspicious region
8. Merchant decides whether to approve the refund
9. Bittensor chain distributes emission: 41% to Miners + 41% to Validators + 18% to subnet owner

Monetization roadmap:
Phase 1 (Months 1-3): Synthetic probes only, build Miner ecosystem
Phase 2 (Months 4-6): Integrate merchant API, charge per query
Phase 3 (Months 6+): Expand to insurance claims, legal forensics, cross-border e-commerce
