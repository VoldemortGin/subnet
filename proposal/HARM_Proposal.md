# HARM: Hey Asshole, Return Money

## A Bittensor Subnet for Decentralized Image Forgery Detection in E-Commerce Refund Fraud

**Subnet Proposal — Bittensor Subnet Ideathon (HackQuest)**

**Authors:** HARM Team  
**Date:** May 2026  
**Version:** 0.1.0

---

## Abstract

HARM is a Bittensor subnet that creates a decentralized marketplace for image forgery detection, targeting the multi-billion-dollar problem of e-commerce refund fraud. On platforms like Taobao, Pinduoduo, and JD.com, dishonest customers exploit "refund-only" policies by submitting Photoshopped or AI-generated photographs of fabricated product damage — keeping the merchandise while pocketing full refunds. HARM deploys a hybrid network of machine learning models and human forensic analysts, incentivized through Bittensor's emission mechanism, to detect traditional image manipulations: copy-move forgery, splicing, JPEG compression mismatch, noise injection, and metadata tampering. Unlike existing subnets such as BitMind (SN34), which focus exclusively on detecting AI-generated images, HARM addresses the far more prevalent category of *human-crafted* forgeries that constitute the majority of real-world fraud. The subnet's design admits both GPU-powered ML miners and human annotators, creating an intelligence market where accuracy — not hardware — determines rewards.

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Problem Statement](#2-problem-statement)
3. [Related Work & Competitive Landscape](#3-related-work--competitive-landscape)
4. [System Architecture](#4-system-architecture)
5. [Mechanism Design](#5-mechanism-design)
6. [Miner Pipeline](#6-miner-pipeline)
7. [Validator Pipeline](#7-validator-pipeline)
8. [Scoring & Incentive Mechanism](#8-scoring--incentive-mechanism)
9. [Anti-Gaming Measures](#9-anti-gaming-measures)
10. [Emission Distribution](#10-emission-distribution)
11. [Go-to-Market Strategy](#11-go-to-market-strategy)
12. [Technical Implementation Status](#12-technical-implementation-status)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)

---

## 1. Introduction & Motivation

The name says it all. HARM — *Hey Asshole, Return Money* — is a direct response to a specific, widespread, and economically devastating category of online fraud. Every day, millions of e-commerce transactions on Chinese platforms are challenged by buyers who claim products arrived damaged, defective, or not-as-described. To support these claims, they submit photographs. Many of those photographs are lies.

The fraud is simple: take a photo of a perfectly good product, open Doubao (豆包), Xingtu (醒图), or Meitu's AI editor, tell the AI to add a scratch, a stain, a crack, or a missing component — then submit the doctored image to the platform's dispute resolution system and walk away with a full refund *and* the product. The explosion of consumer-grade AI editing tools in 2025-2026 has democratized image forgery: what once required Photoshop expertise now takes a single prompt and five seconds. At scale, this costs merchants billions of yuan annually. Pinduoduo's "refund-only" policy, introduced to protect consumers, has been weaponized by bad actors into an industrial-scale grift.

The core insight behind HARM is that this is fundamentally an *image forensics* problem — and image forensics is exactly the kind of task that benefits from decentralized, competitive intelligence markets. No single model catches every forgery. No single human can review millions of images per day. But a network of diverse analysts — some running PSCC-Net on a laptop, some running TruFor on a GPU cluster, some scrutinizing metadata with the naked eye — competing for accuracy under Bittensor's incentive mechanism? That scales.

---

## 2. Problem Statement

### 2.1 The Refund Fraud Epidemic

China's e-commerce market processed over 15 trillion yuan (~$2.1 trillion USD) in transactions in 2025. Major platforms including Taobao, Pinduoduo, JD.com, and Douyin Shop have adopted consumer-friendly "refund-only" (仅退款) policies that allow buyers to receive refunds without returning merchandise. While these policies reduce friction for legitimate complaints, they have created a perverse incentive: dishonest buyers fabricate evidence of product damage to obtain free goods.

The fraud workflow is well-documented:

1. Buyer receives product in good condition.
2. Buyer opens Doubao (豆包), Xingtu (醒图), or similar AI editing app and prompts: "add a crack to this screen" or "make this product look water-damaged."
3. AI generates a realistic edit in seconds — no technical skill required.
4. Buyer submits the AI-doctored photo through the platform's dispute system.
5. Platform's automated or manual review fails to detect the forgery.
6. Buyer receives refund. Buyer keeps product. Merchant absorbs the loss.

Conservative industry estimates place annual losses from image-based refund fraud at 10-50 billion yuan ($1.4-7 billion USD). Individual merchants report fraud rates of 5-15% on refund claims.

### 2.2 Why Current Solutions Fail

**Manual review does not scale.** Platforms employ customer service teams to adjudicate disputes, but the volume of claims (millions per day across major platforms) makes thorough image forensic analysis of every submission impossible. Reviewers spend seconds per image and rely on gut instinct.

**Existing ML solutions are narrow.** Commercial image forensics tools are either (a) proprietary, expensive, and unavailable to small merchants, or (b) trained on academic datasets that poorly represent the specific manipulations used in e-commerce fraud (product photos, mobile phone captures, heavy JPEG compression).

**AI-detection subnets miss the target.** Bittensor's BitMind (Subnet 34) detects whether an image was *fully generated* by AI (e.g., Midjourney, DALL-E). But refund fraud images are not fully AI-generated — they are real photographs with *localized AI-assisted edits* (a crack added by Doubao, a stain painted by Xingtu). BitMind's NPR/UCF models, designed to detect whole-image AI generation artifacts, are blind to these partial, AI-tool-assisted manipulations applied to otherwise authentic photos.

### 2.3 The Opportunity

HARM addresses this gap by creating a decentralized, incentivized marketplace for image forgery detection that:

- Covers the **full fraud spectrum**: AI-assisted local edits (Doubao adding a crack), AI inpainting (Xingtu replacing a region), fully AI-generated fake damage photos (Doubao generating from scratch), and reused/stolen refund images — not just one category
- Admits both ML models and human analysts, creating a hybrid intelligence market where contextual judgment (wrong product model, impossible lighting, mismatched background) complements pixel-level forensics
- Focuses on a concrete, monetizable vertical (e-commerce fraud prevention) rather than generic "is this image real?" classification
- Operates with lightweight validator hardware requirements, lowering the barrier to participation

---

## 3. Related Work & Competitive Landscape

### 3.1 BitMind (Bittensor Subnet 34)

BitMind is the closest existing subnet to HARM's domain. A direct comparison clarifies the differentiation:

| Dimension | BitMind (SN34) | HARM |
|-----------|---------------|------|
| **Detection target** | Fully AI-generated images (deepfakes, diffusion outputs) | Full fraud spectrum: AI-assisted edits, AI inpainting, fully AI-generated, and reused images |
| **Core models** | NPR, UCF (AI upsampling artifact detection) | 7 backends: ELA, ManTraNet, TruFor, CAT-Net, MVSS-Net, PSCC-Net, FOCAL |
| **Miner type** | Model weight submission only | ML models + human annotators + hybrid |
| **Validator hardware** | 8x A100 GPUs required | CPU-capable (image manipulation scripts + scoring) |
| **Forgery methods detected** | GAN/diffusion artifacts | Copy-move, splicing, compression mismatch, noise injection, metadata forgery |
| **Output** | Binary (real/fake) | Structured report: verdict, confidence, mask, method classification |
| **Use case** | Generic deepfake detection | E-commerce refund fraud prevention |

The two subnets are complementary, not competitive. BitMind answers "Was this entire image generated by AI?" HARM answers "Was this real photograph locally edited using AI tools?" In the e-commerce fraud context, the latter question is far more relevant — fraudsters use Doubao or Xingtu to add fake damage to real product photos, producing images that are 95% authentic and 5% AI-manipulated. BitMind's whole-image classification sees the 95% authentic signal; HARM's localization pipeline catches the 5% that was forged.

### 3.2 Academic Image Forensics

The image forensics research community has produced numerous models for manipulation detection. HARM integrates seven established architectures as miner backends:

- **Mesorch** (AAAI 2025): Multi-scale parallel Transformer + CNN architecture. The current state-of-the-art, surpassing all prior methods with fewer parameters (62.2M) and 3.6× lower FLOPs than TruFor. Avg AUC 0.943, Avg IoU 0.636.
- **TruFor** (CVPR 2023): Transformer-based forensic model combining RGB and learned noise fingerprints (Noiseprint++). Former SOTA with strong cross-dataset generalization. 68.7M parameters.
- **FOCAL** (TMLR 2023): Forgery contrastive learning with unsupervised clustering at test time. Exceptional IoU performance (+24.8% on Coverage, +15% on CASIA over prior methods). ~150M parameters.
- **IML-ViT** (2023): Vision Transformer with MAE pretraining for manipulation localization. Achieves the highest standardized F1 (0.725) under IMDL-BenCo protocol. ~89M parameters.
- **ProFact** (2024): Progressive feedback-enhanced transformer. Best generalization across 9 public forensic datasets. ~70M parameters.
- **PSCC-Net** (TCSVT 2022): Progressive spatio-channel correlation network. Ultra-lightweight at ~2M parameters, 0.019s/image inference — ideal for resource-constrained miners.
- **ManTraNet** (CVPR 2019): End-to-end manipulation tracing network. Lightweight baseline at ~3.8M parameters, CPU-capable.
- **ELA (Error Level Analysis):** Classical JPEG compression artifact analysis. Zero external dependencies, useful as a fallback baseline.

No single model dominates across all forgery types, which is precisely why a competitive marketplace that rewards accuracy — regardless of method — produces better aggregate results than any single-model deployment.

---

## 4. System Architecture

HARM follows the standard Bittensor subnet architecture with domain-specific components:

```
                          ┌──────────────────────────────┐
                          │      Bittensor Network       │
                          │   Yuma Consensus + Emissions │
                          └──────────┬───────────────────┘
                                     │
                          Weight Submission │ Emission Distribution
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
   ┌────────▼────────┐    ┌─────────▼─────────┐    ┌────────▼────────┐
   │  Validator #1   │    │   Validator #2     │    │  Validator #N   │
   │                 │    │                    │    │                 │
   │ ┌─────────────┐ │    │  ┌─────────────┐  │    │ ┌─────────────┐ │
   │ │Forge Engine │ │    │  │Forge Engine │  │    │ │Forge Engine │ │
   │ │(Probe Gen)  │ │    │  │(Probe Gen)  │  │    │ │(Probe Gen)  │ │
   │ └──────┬──────┘ │    │  └──────┬──────┘  │    │ └──────┬──────┘ │
   │        │        │    │         │         │    │        │        │
   │ ┌──────▼──────┐ │    │  ┌──────▼──────┐  │    │ ┌──────▼──────┐ │
   │ │Task Distrib.│ │    │  │Task Distrib.│  │    │ │Task Distrib.│ │
   │ └──────┬──────┘ │    │  └──────┬──────┘  │    │ └──────┬──────┘ │
   │        │        │    │         │         │    │        │        │
   │ ┌──────▼──────┐ │    │  ┌──────▼──────┐  │    │ ┌──────▼──────┐ │
   │ │  Scorer     │ │    │  │  Scorer     │  │    │ │  Scorer     │ │
   │ └─────────────┘ │    │  └─────────────┘  │    │ └─────────────┘ │
   └────────┬────────┘    └─────────┬─────────┘    └────────┬────────┘
            │                       │                       │
            │          Task Requests (Images)               │
            │         ◄─────────────┼──────────────►        │
            │                       │                       │
   ┌────────▼────────┐    ┌─────────▼─────────┐    ┌────────▼────────┐
   │   Miner #1      │    │    Miner #2       │    │   Miner #M      │
   │  (TruFor GPU)   │    │  (Human Analyst)  │    │ (PSCC-Net CPU)  │
   │                 │    │                    │    │                 │
   │ ┌─────────────┐ │    │  ┌─────────────┐  │    │ ┌─────────────┐ │
   │ │Model Select │ │    │  │Manual Review│  │    │ │Model Select │ │
   │ └──────┬──────┘ │    │  └──────┬──────┘  │    │ └──────┬──────┘ │
   │ ┌──────▼──────┐ │    │  ┌──────▼──────┐  │    │ ┌──────▼──────┐ │
   │ │  Detection  │ │    │  │  Annotation │  │    │ │  Detection  │ │
   │ └──────┬──────┘ │    │  └──────┬──────┘  │    │ └──────┬──────┘ │
   │ ┌──────▼──────┐ │    │  ┌──────▼──────┐  │    │ ┌──────▼──────┐ │
   │ │  Response   │ │    │  │  Response   │  │    │ │  Response   │ │
   │ └─────────────┘ │    │  └─────────────┘  │    │ └─────────────┘ │
   └─────────────────┘    └───────────────────┘    └─────────────────┘
```

### 4.1 Validator Node

The validator is deliberately lightweight. It requires no GPU and performs three functions:

1. **Forge Engine**: Generates probe tasks by applying known tampering operations (copy-move, splicing, compression mismatch, noise injection) to clean source images. Each probe has a deterministic ground truth mask.
2. **Task Distribution**: Dispatches a mix of probe tasks (30%) and real merchant queries (70%) to miners, managing the commit-reveal protocol.
3. **Scorer**: Evaluates miner responses against ground truth (for probes) and weighted majority vote (for consensus tasks), computing per-epoch scores.

### 4.2 Miner Node

Miners are method-agnostic "Image Forensic Analysts." The subnet does not prescribe how a miner reaches its conclusion — only that the conclusion is accurate. A miner may use:

- Any of the 7 supported ML backends
- Manual human analysis
- A hybrid pipeline (ML pre-filter + human review)
- Custom models not in the registry

Output format is standardized: `{verdict, confidence, mask, method}`.

### 4.3 Digital Commodity

The digital commodity produced by HARM is an **image forgery detection report** containing:

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | enum | `AUTHENTIC` or `TAMPERED` |
| `confidence` | float [0, 1] | Model/analyst confidence in the verdict |
| `mask` | ndarray (H x W) | Binary pixel mask indicating tampered regions (255 = tampered) |
| `method` | enum \| null | Forgery method classification: `COPY_MOVE`, `SPLICING`, `COMPRESSION`, `INPAINTING`, `METADATA` |
| `latency_ms` | float | Response time in milliseconds |

---

## 5. Mechanism Design

### 5.1 Task Composition

Each validation epoch distributes a mixed workload:

- **Probe tasks (30%)**: Validator-generated fakes with known ground truth. These are the objective quality signal. The validator takes a clean image, applies a forgery operation via the Forge Engine, and records the exact manipulation mask.
- **Consensus tasks (70%)**: Real-world images submitted by merchant clients (or, during bootstrapping, drawn from public forensics datasets). These have no ground truth; quality is assessed via weighted majority vote.

The 70/30 split balances two needs: (a) sufficient probe volume for reliable miner ranking, and (b) majority real-world workload so the subnet produces economically useful output.

### 5.2 Forge Engine

The validator's Forge Engine (`src/validator/forge.py`) generates probe tasks using four tampering methods:

**Copy-Move.** A rectangular region is duplicated within the same image. The source patch is copied to a non-overlapping destination location. This simulates the common fraud technique of duplicating a defect to make damage appear more extensive.

**Splicing.** A region from a donor image (or synthetic noise patch) is composited into the target image with alpha blending (alpha = 0.85). This simulates the insertion of damage from a different photograph.

**Compression Mismatch.** A rectangular region is re-compressed at a different JPEG quality level (Q = 10-35) and pasted back. The resulting compression artifact boundary is invisible to the human eye but detectable by forensic analysis.

**Noise Injection.** Localized Gaussian noise (sigma = 15-40) is added to a rectangular region, simulating the noise inconsistencies left by inpainting or clone-stamp operations.

Each probe records the exact binary mask of the tampered region and the method used, providing deterministic ground truth for scoring.

### 5.3 Commit-Reveal Protocol

To prevent miners from free-riding by copying other miners' responses, HARM employs a two-phase commit-reveal protocol:

**Phase 1 — Commit.** Upon receiving a task, each miner computes their response, generates a random nonce, and submits `hash(response || nonce)` to the validator. This cryptographic commitment is binding — the miner cannot change their answer after seeing others' commitments.

**Phase 2 — Reveal.** After all miners have submitted their commitment hashes, the validator opens the reveal window. Each miner submits their actual `(response, nonce)` pair. The validator verifies that `hash(response || nonce)` matches the committed hash. Mismatches result in a zero score for the epoch.

This ensures every miner's response is independently generated before any miner can observe others' answers.

### 5.4 Difficulty Adaptation

The Forge Engine implements adaptive difficulty to prevent the network from plateauing:

1. **Monitoring**: The validator tracks network-wide probe accuracy over a rolling window.
2. **Trigger**: When average accuracy exceeds 90% for 3 consecutive tempos (epochs), difficulty increases.
3. **Escalation methods**:
   - Reduce tampered region size (`min_frac` and `max_frac` parameters decrease)
   - Introduce AI-assisted inpainting (seamless clone, Poisson blending) for more realistic forgeries
   - Mix multiple forgery methods within a single image
   - Apply post-processing (JPEG re-compression, rescaling) to destroy simple forensic traces
4. **Floor**: Difficulty never decreases, ensuring continuous improvement pressure on miners.

---

## 6. Miner Pipeline

### 6.1 Method-Agnostic Design

HARM miners are not required to use any specific model or technique. The subnet's protocol (`src/protocol.py`) defines only the input (image) and output (verdict, confidence, mask, method) — not the mechanism by which the miner arrives at its conclusion. This is a deliberate design choice that:

- Allows human analysts to compete alongside ML models
- Encourages methodological diversity (ensemble approaches, custom models, hybrid pipelines)
- Prevents the subnet from becoming a benchmark for a single architecture

### 6.2 Supported Model Backends

The HARM codebase provides a model registry (`src/miner/model_registry.py`) with seven pre-integrated backends, plus a built-in ELA+noise analysis pipeline that requires no external model weights:

| Model | Venue | Parameters | Min VRAM | CPU? | Avg AUC | Avg F1 | Avg IoU |
|-------|-------|-----------|----------|------|---------|--------|---------|
| **Mesorch-P** | AAAI 2025 | 62.2M | 4-6 GB | Barely | **0.943** | **0.676** | **0.636** |
| **TruFor** | CVPR 2023 | 68.7M | 4-6 GB | Barely | 0.924 | 0.627 | 0.586 |
| **FOCAL** | TMLR 2023 | ~150M | 8-10 GB | No | 0.914 | — | 0.728 |
| **IML-ViT** | 2023 | ~89M | 6-8 GB | Barely | 0.862 | 0.725† | — |
| **ProFact** | 2024 | ~70M | 4-6 GB | Barely | — | Best generalization | — |
| PSCC-Net | TCSVT 2022 | ~2M | ~2 GB | **Yes** | 0.894 | 0.577 | 0.478 |
| ManTraNet | CVPR 2019 | ~3.8M | ~2 GB | **Yes** | 0.876 | 0.295 | — |
| Human Analyst | N/A | N/A | N/A | N/A | Variable | Variable | Variable |

*†IML-ViT F1 measured under IMDL-BenCo standardized evaluation protocol. Top 5 SOTA models shown in bold.*

**Recommendation for new miners:** Resource-constrained miners should start with PSCC-Net (~2M params, CPU-capable, 0.019s/image). Mid-tier GPU miners (4-6 GB VRAM) should run **Mesorch-P** (AAAI 2025, highest accuracy at lowest compute cost among SOTA models) or **TruFor**. High-end miners (8+ GB VRAM) can run **FOCAL** or ensemble multiple models for maximum accuracy.

### 6.3 Human Miners

HARM explicitly supports human miners — individuals who manually inspect images and produce forensic annotations. This is unusual for a Bittensor subnet and is a key differentiator. Rationale:

- **Complementary expertise**: Trained human analysts catch manipulations that current ML models miss (semantic inconsistencies, shadow/lighting errors, contextual implausibility).
- **Cold start**: Human miners provide baseline quality before ML miners have fine-tuned their models.
- **Adversarial robustness**: Attackers who craft adversarial examples to fool ML models cannot simultaneously fool human reviewers.

The scoring formula's 5% latency weight (Section 8) is intentionally low to ensure human miners — whose response times are measured in minutes rather than milliseconds — remain competitive on accuracy.

### 6.4 Miner Response Format

All miners produce a standardized `MinerResponse`:

```
MinerResponse {
    task_id:      str              -- Unique task identifier
    verdict:      AUTHENTIC | TAMPERED
    confidence:   float [0.0, 1.0] -- Calibrated confidence
    method:       ForgeryMethod?   -- Optional method classification
    mask:         ndarray?         -- Binary pixel mask (H x W)
    latency_ms:   float            -- Time to produce response
}
```

---

## 7. Validator Pipeline

### 7.1 Image Sources

Validators source images from two channels:

**Merchant Queries (target: 70% of tasks).** Real-world images submitted by e-commerce merchants for forensic analysis. During the bootstrapping phase (before merchant partnerships are established), validators substitute images from public forensics datasets (CASIA v2.0, Columbia, NIST MFC, Coverage).

**Probe Tasks (30% of tasks).** Synthetically tampered images generated by the validator's Forge Engine from clean originals. These provide the objective ground truth signal that anchors the scoring system.

### 7.2 Validation Epoch Flow

```
1. Validator selects images for the epoch
   ├── 70%: Real queries (or dataset images during bootstrap)
   └── 30%: Clean images → Forge Engine → Tampered images + ground truth masks

2. Task distribution with commit-reveal
   ├── Phase 1: Broadcast task images to all miners
   │             Miners compute responses, submit hash(response || nonce)
   └── Phase 2: After all hashes collected, open reveal window
                 Miners submit (response, nonce), validator verifies hashes

3. Scoring
   ├── Probe tasks: Score against ground truth (Section 8.1)
   ├── Consensus tasks: Weighted majority vote (Section 8.2)
   └── Latency: Penalize slow responses (Section 8.3)

4. Epoch score aggregation → Weight submission to Bittensor chain
```

### 7.3 Lightweight Validator Requirements

A critical design advantage of HARM over other image-focused subnets: **validators require no GPU**. The Forge Engine uses OpenCV operations (region copying, JPEG encode/decode, Gaussian noise generation) that run efficiently on CPU. The Scorer performs simple arithmetic (IoU computation, confidence calibration, weighted voting). This dramatically lowers the barrier to becoming a validator, improving decentralization.

Minimum validator requirements:
- 4-core CPU
- 8 GB RAM
- 50 GB storage (for image datasets)
- Stable network connection

Compare this to BitMind (SN34), which requires 8x A100 GPUs for validators — a setup costing $100,000+ in hardware.

---

## 8. Scoring & Incentive Mechanism

### 8.1 Probe Score (Ground Truth Known)

For probe tasks where the validator knows the exact ground truth, each miner's response is scored as:

```
S_probe = P_correct * (alpha * IoU + beta * C_cal + gamma * M_bonus)
```

Where:

- **P_correct** = 1 if the miner's verdict matches ground truth, 0 otherwise. This is a *gate*: an incorrect verdict zeroes the entire score regardless of other metrics.

- **IoU** = Intersection over Union between the miner's predicted mask and the ground truth mask:
  ```
  IoU = |pred_mask AND gt_mask| / |pred_mask OR gt_mask|
  ```

- **C_cal** = Confidence calibration score, rewarding well-calibrated confidence values:
  ```
  C_cal = 1 - |confidence - ideal|
  ```
  where `ideal = 1.0` for tampered images and `ideal = 0.0` for authentic images.

- **M_bonus** = 1.0 if the miner correctly identifies the forgery method, 0.0 otherwise.

- **Weights**: alpha = 0.4, beta = 0.4, gamma = 0.2

This formula rewards three distinct capabilities: spatial localization (IoU), calibration (C_cal), and forensic classification (M_bonus). The verdict gate ensures that miners cannot score well by producing high-confidence wrong answers with good masks.

### 8.2 Consensus Score (No Ground Truth)

For real-world merchant queries where no ground truth exists, HARM uses a weighted majority vote:

```
majority_verdict = argmax_v { SUM_i [ w_i * I(verdict_i == v) ] }
```

Where `w_i` is miner *i*'s historical probe accuracy — their average probe score over a rolling window. This means a miner's vote weight in consensus tasks is earned through demonstrated accuracy on objective probes.

A miner scores `S_consensus = 1.0` if their verdict matches the weighted majority, and `S_consensus = 0.0` otherwise.

### 8.3 Latency Score

```
S_latency = max(0, 1 - t_response / t_max)
```

Where `t_max` is the task timeout (default: 30 seconds). Faster responses score higher, but this component is weighted at only 5% of the total score — deliberately low to keep human miners competitive.

### 8.4 Epoch Aggregation

The total score for a miner in an epoch is:

```
S_total = omega_probe * avg(S_probe) + omega_consensus * avg(S_consensus) + omega_latency * S_latency
```

With weights:
- omega_probe = 0.60 (probe accuracy dominates)
- omega_consensus = 0.35 (consensus alignment matters)
- omega_latency = 0.05 (speed is a tiebreaker, not a primary criterion)

These weights are implemented in `src/validator/scorer.py` as the `ScoringWeights` dataclass and are tunable by subnet governance.

---

## 9. Anti-Gaming Measures

A scoring mechanism is only as good as its resistance to manipulation. HARM implements six specific countermeasures:

### 9.1 Sybil Attacks

**Threat**: An attacker registers multiple miner identities to amplify their voting weight in consensus tasks.

**Mitigation**: 
- Consensus voting is weighted by *probe accuracy*, not by identity count. Sybil identities that perform poorly on probes contribute negligible weight.
- Bittensor's registration cost (TAO stake) makes large-scale Sybil attacks economically prohibitive.
- The 60% weight on probe scores (which are objective and per-identity) ensures that duplicating identities without improving accuracy yields no benefit.

### 9.2 Free-Riding / Response Copying

**Threat**: A miner waits for other miners to respond, copies the majority answer, and submits it as their own.

**Mitigation**: The commit-reveal protocol (Section 5.3) makes this impossible. All miners must commit their hashed response before any responses are revealed. A miner cannot see others' answers until after their own answer is cryptographically locked.

### 9.3 Always-Say-Fake Strategy

**Threat**: A miner labels every image as `TAMPERED` to maximize recall on tampered probes.

**Mitigation**: The probe set includes authentic (unmodified) images. The true/fake ratio in probes is *dynamic and non-public* — validators randomize the proportion each epoch. An "always-say-fake" miner will consistently fail on authentic probes, receiving `P_correct = 0` and a zeroed score.

### 9.4 Always-Say-Real Strategy

**Threat**: Conversely, a miner labels every image as `AUTHENTIC`.

**Mitigation**: Same mechanism — the dynamic true/fake ratio ensures this strategy fails on tampered probes. Additionally, since consensus tasks are weighted by probe accuracy, an always-real miner's consensus votes carry near-zero weight.

### 9.5 Model Overfitting

**Threat**: A miner overfits to the validator's specific forgery generation patterns, performing well on probes but poorly on real-world images.

**Mitigation**:
- Multiple forgery methods are used, randomized per probe.
- Difficulty adaptation (Section 5.4) continuously introduces new manipulation techniques.
- The 35% consensus weight rewards performance on real-world images (which differ from synthetic probes).
- Validators source clean images from diverse datasets, preventing memorization.

### 9.6 Validator Collusion

**Threat**: A validator conspires with specific miners, feeding them probe answers in advance.

**Mitigation**:
- Multiple validators independently generate probes and score miners. A colluding validator-miner pair represents a single signal in a multi-validator system.
- Bittensor's Yuma Consensus mechanism detects anomalous weight patterns — if one validator's scores diverge dramatically from others, its weights are discounted.
- Cross-verification: validators can optionally share probe ground truths post-epoch to audit each other's scoring.

---

## 10. Emission Distribution

HARM follows the standard Bittensor emission split:

| Recipient | Share | Rationale |
|-----------|-------|-----------|
| Subnet Owner | 18% | Development, maintenance, merchant partnerships |
| Miners | 41% | Incentive for accurate forensic analysis |
| Validators / Stakers | 41% | Incentive for probe generation, scoring, and stake |

This distribution is set by Bittensor protocol defaults and may be adjusted through subnet governance as the network matures.

---

## 11. Go-to-Market Strategy

### 11.1 Phase 1: Bootstrapping (Months 1-3)

**Objective**: Build miner base, establish baseline accuracy, prove mechanism viability.

- **Miner acquisition**: Target Bittensor community miners with existing GPU hardware. HARM's lightweight computational requirements (PSCC-Net runs on CPU; TruFor needs only 4-6 GB VRAM) enable "dual mining" — miners can participate in HARM alongside compute-intensive subnets.
- **Task source**: 100% synthetic probe tasks from public forensics datasets (CASIA v2.0, Columbia, Coverage, NIST MFC).
- **Milestone**: Achieve >85% network accuracy on diverse forgery types; demonstrate commit-reveal protocol at scale.

### 11.2 Phase 2: Merchant Partnerships (Months 4-6)

**Objective**: Generate revenue through real-world forensic queries.

- **Integration targets**: E-commerce tool providers — ERP systems (Jushita, Wangjiangjia), customer service platforms (Xiaoneng, Meiqia), and dispute management tools used by Taobao/Pinduoduo/JD merchants.
- **API product**: REST API accepting suspicious refund photographs, returning structured forensic reports (verdict, confidence, mask, method). Pricing: per-query or monthly subscription.
- **Merchant value proposition**: "Automated second opinion on refund claims. Reduce fraudulent refund losses by 30-60%."
- **Transition**: Move from 100% synthetic to 70% real merchant queries + 30% probes.

### 11.3 Phase 3: Enterprise & Beyond (Months 6+)

**Objective**: Expand to adjacent verticals.

- **Insurance**: Fraud claim verification (auto insurance damage photos, property damage claims).
- **Legal/forensic**: Evidence authenticity verification for litigation support.
- **Content platforms**: User-generated content authenticity checks (product reviews, social media).
- **Cross-border e-commerce**: International platforms (Temu, Shein, AliExpress) facing similar refund fraud challenges.

### 11.4 Why Bittensor?

The question "Why build this as a Bittensor subnet rather than a centralized service?" has four concrete answers:

1. **Human-compute hybrid coordination.** HARM's unique allowance for human miners alongside ML models creates a labor marketplace that is difficult to manage centrally — recruitment, quality control, payment, task assignment — but trivial on Bittensor, where the incentive mechanism handles all of this automatically.

2. **Built-in quality control.** The probe scoring mechanism provides continuous, objective quality assessment of every participant. Underperforming miners are automatically de-incentivized without manual intervention.

3. **Global workforce, zero management overhead.** Anyone in the world can become a HARM miner by registering on the subnet. No hiring, no contracts, no payroll. The network scales with demand.

4. **Continuous improvement through competition.** Miners are in constant competition for emissions. This creates natural pressure to improve models, find better techniques, and maintain high accuracy — without any central team managing an optimization roadmap.

---

## 12. Technical Implementation Status

The HARM codebase is functional and demonstrates end-to-end feasibility:

| Component | Status | Location |
|-----------|--------|----------|
| Protocol layer (data types, enums, message formats) | Complete | `src/protocol.py` |
| Validator Forge Engine (4 tampering methods, ground truth mask generation) | Complete | `src/validator/forge.py` |
| Validator Scorer (probe scoring, consensus voting, epoch aggregation) | Complete | `src/validator/scorer.py` |
| Miner Detector (ELA + noise analysis built-in pipeline) | Complete | `src/miner/detector.py` |
| Model Registry (7 backend integrations, auto-selection) | Complete | `src/miner/model_registry.py` |
| Backend Interface (abstract base class for detection backends) | Complete | `src/miner/backends/base.py` |
| Backend Implementations (ELA, ManTraNet, TruFor, CAT-Net, MVSS-Net, PSCC-Net, FOCAL) | Complete | `src/miner/backends/*.py` |
| Demo Pipeline (end-to-end probe generation, detection, scoring) | Complete | `demo/run_demo.py` |

**Not yet implemented** (planned for mainnet):
- Commit-reveal protocol (network layer)
- Bittensor chain integration (axon/dendrite, weight submission)
- Difficulty adaptation controller
- Merchant API gateway
- Human miner web interface

---

## 13. Conclusion

HARM fills a specific, unaddressed gap in the Bittensor ecosystem. While BitMind (SN34) asks "Was this image generated by AI?", HARM asks the more commercially relevant question: "Was this photograph tampered with?" The distinction matters because the vast majority of e-commerce refund fraud involves real photographs with localized human edits — not fully synthetic images.

The subnet's design reflects three convictions:

1. **Accuracy beats speed.** The scoring formula weights accuracy (95%) far more than latency (5%), creating a market where careful analysis — including human analysis — is rewarded over fast guessing.

2. **Diversity beats uniformity.** By supporting 7 ML backends and human annotators in a method-agnostic framework, HARM ensures that the network's aggregate intelligence exceeds any single model's capability.

3. **Lightweight infrastructure broadens participation.** CPU-capable validators and sub-2GB miner models lower the barrier to entry, improving decentralization and reducing the plutocratic advantage of GPU-rich operators.

The name is aggressive. The problem is real. The mechanism is sound. Whether the fraud image was locally touched up in Doubao, inpainted in Xingtu, generated from scratch by AI, or stolen from someone else's refund claim — HARM catches it. Merchants deserve better than to be robbed by a five-second AI prompt. HARM is here to make that expensive for the assholes.

---

## 14. References

1. Wu, Y., AbdAlmageed, W., & Natarajan, P. (2019). ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries with Anomalous Features. *CVPR 2019*.
2. Guillaro, F., Cozzolino, D., Sud, A., Dufour, N., & Verdoliva, L. (2023). TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization. *CVPR 2023*.
3. Kwon, M. J., Yu, I. J., Nam, S. H., & Lee, H. K. (2022). CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing. *WACV 2022*.
4. Chen, X., Dong, C., Ji, J., Cao, J., & Li, X. (2021). Image Manipulation Detection by Multi-View Multi-Scale Supervision. *ICCV 2021*.
5. Liu, X., Liu, Y., Chen, J., & Liu, X. (2022). PSCC-Net: Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization. *IEEE TCSVT*.
6. Wu, J., Shi, Z., & Zhan, S. (2023). FOCAL: A Forgery Contrastive Learning Framework for Image Forgery Localization.
7. Bittensor Foundation. (2024). Bittensor Whitepaper. https://bittensor.com/whitepaper
8. BitMind. (2024). Subnet 34: AI Image Detection. https://github.com/bitmind-ai

---

*HARM: Because "the customer is always right" should not extend to Photoshop.*
