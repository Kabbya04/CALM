# Novelty positioning: CALM (v1 & v2)

Brief positioning note for the NeurIPS submission. *Web search was used for landscape context (training-free CLIP adapters, selective prediction for VLMs); detailed claims should stay aligned with your experiments and related-work citations in the paper.*

---

## One-sentence pitch

CALM treats **frozen VLMs + episodic memory + prototypical decisions + calibrated deployment gating (and optional human feedback)** as a **single system**, and studies a **second variant (v2)** that adds **multi-encoder confidence** and **hierarchical compressed memory**, with **matched ImageNet episodic and continual protocols** and an explicit discussion of **representation consistency** when fusing encoders.

---

## What nearby lines of work typically do (landscape)

| Area | Typical focus | Common gap relative to CALM |
|------|----------------|-----------------------------|
| **Training-free CLIP adaptation** (e.g. [Tip-Adapter](https://arxiv.org/abs/2207.09519), [Proto-Adapter](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175357/)) | Strong **few-shot** accuracy via caches/prototypes/adapters **without** fine-tuning the trunk | Usually **batch** few-shot evaluation; **not** framed as a continual stream with explicit memory budgets; **no** integrated **deploy vs request-feedback** policy tied to the same prototype pipeline |
| **Continual learning with foundation models** | Prompt tuning, adapters, replay in feature or parameter space ([DualPrompt](https://arxiv.org/abs/2203.06999)-style methods, latent replay surveys) | Often **updates internal representations** or uses **parametric heads**; CALM keeps the **encoder frozen** and adapts **non-parametrically** via memory + prototypes |
| **Selective prediction / abstention for VLMs** (e.g. [ReCoVERR-style reasoning](https://arxiv.org/abs/2402.15610), NeurIPS posters on selective VLMs) | When to answer vs abstain, sometimes with extra reasoning or retrieval | Frequently **decoupled** from a **minimal episodic few-shot + continual** engine built **only** on frozen embeddings and class prototypes; CALM’s **readiness agent** is wired **directly** to the same similarity scores as classification |

*These are coarse contrasts for positioning, not exhaustive literature claims.*

---

## Distinctive elements to emphasize (your contributions)

1. **Integrated deployment story**  
   Non-parametric adaptation is combined with **temperature-calibrated confidence** and a **binary deploy / request human label** policy, with **immediate memory updates** from corrections—framed as one loop rather than a separate “calibration paper” plus a “few-shot paper.”

2. **Continual + few-shot under one frozen-VLM design**  
   Same backbone constraint (\(\nabla_\theta \mathcal{L} = 0\)), **episodic** support, and **continual** evaluation on a **fixed protocol** (ImageNet splits you report), comparable to **standard tabulated baselines** in the submission.

3. **CALM-v2 as a systems-oriented extension**  
   **Ensemble disagreement** across CLIP variants for richer confidence, plus **tiered compressed memory** for **capacity**, with an honest **empirical** comparison to v1 and a **plausible mechanism** (cross-encoder / prototype–query consistency) when scores drop.

4. **Actionable ablation direction**  
   **Single-backbone v2** (hierarchical memory retained, scoring in one encoder space) to **separate** fusion effects from memory design—useful for reviewers who ask “what if the gap is just multi-encoder scoring?”

---

## How to phrase novelty diplomatically (reviewer-friendly)

- Prefer: *“In our evaluated setting, CALM-v1 compares favorably on few-shot accuracy to the listed CLIP-centric baselines; v2 remains in a similar range while targeting deployment-oriented signals and memory growth.”*  
- Avoid: *“We beat all prior methods”* unless every row is strictly comparable (same backbone, protocol, and compute).

---

## Caveats to keep in mind

- **Individual ingredients** (prototypes, temperature scaling, CLIP, rehearsal) are not new; **novelty is the combination, evaluation, and v1↔v2 analysis** under **matched protocols**.  
- If any **internal report / thesis / arXiv** exists with overlapping text, align venue statements with **actual prior disclosure** to avoid self-plagiarism or dual-submission issues.

---

## References (from web search)

- Zhang et al., *Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling*, ECCV 2022. <https://arxiv.org/abs/2207.09519>  
- Li et al., *Proto-Adapter: Efficient Training-Free CLIP-Adapter for Few-Shot Image Classification* (PMC). <https://pmc.ncbi.nlm.nih.gov/articles/PMC11175357/>  
- Wang et al., *DualPrompt: Complementary Prompting for Rehearsal-Free Continual Learning*, ECCV 2022. <https://arxiv.org/abs/2203.06999>  
- Chen et al., *Selective “Selective Prediction”: Reducing Unnecessary Abstention in Vision-Language Reasoning*, arXiv:2402.15610. <https://arxiv.org/abs/2402.15610>  

---

*Generated to support drafting; replace or tighten bullets after your final related-work pass.*
