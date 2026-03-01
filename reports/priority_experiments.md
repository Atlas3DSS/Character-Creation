# Priority Experiments: Ship in Days, Not Weeks

**Date**: 2026-02-27
**Source**: External adviser feedback + Codex + Gemini analysis

---

## 3 Quick Shipping Experiments (This Week)

### Experiment 1: Sarcasm-Polite Orthogonal Decomposition ✅ READY
- **Script**: `orthogonal_sarcasm_steering.py`
- **Status**: Purified vectors computed, auto-launches on dev server after arena
- **Prediction**: strong_sarcasm_rate 0.75 → ~1.0 without math degradation
- **8B L29-L30**: cos(sarc,polite) 0.24-0.33 → 0.00 after Gram-Schmidt
- **Time**: ~2 hours

### Experiment 2: Push-Pull Pair Identification (dims 755/2455) 🔍 ANALYZED
- **Finding**: Expressiveness ↔ Formality toggle switch
- **dim 755**: Surprise(z=+7.28), Analytical(+1.22), Polite(+1.01)
- **dim 2455**: anti-Surprise(z=-6.11), Formal(+1.21), Identity(+1.13), Brief(+0.90)
- **Anti-correlation peaks L36-L40** (cos = -0.61)
- **Next**: Max-activating prompt analysis on 27B
- **Time**: ~1 hour once model available

### Experiment 3: Magnitude-Calibrated Alpha ✅ DATA COMPLETE
- **Status**: Full-rank spectral analysis COMPLETE (10K samples × 64 layers × 5120 dims)
- **Runtime**: 21 min GPU SVD + 14.5 min CPU rSVD+LW + 5 min assembly = 40 min total
- **Data**: `fullrank_spectral/fullrank_spectral_report.json` (102 KB)
- **Calibrated alphas**: `fullrank_spectral/calibrated_alphas.json`
- **Key findings**:
  - Eigenvalue growth L0→L63: **1.6M× (math), 3.1M× (sarcasm)**
  - Sarcasm 3-4x larger eigenvalues than math at every layer
  - **Zero math-sarc intrusion from L7+** (max alignment 0.29). Only L0-L6 overlap.
  - Sarcasm eff_dim: 22 mean (vs math 10). k90_sarc doubles L48→L63 (130→268)
  - **Calibrated α curve (ref=L50@α=8)**: L48=12, L52=4.3, L55=2.3, L60=1.3, L63=0.75
  - Current uniform α=8 is **10x too strong at L63** — explains late-layer math degradation
- **Formula**: α_layer = α_base × (ref_median / layer_median), clamped [0.1, 100]
- **Next**: Implement calibrated multi-layer steering sweep on 27B
- **Time**: ~4 hours to implement + sweep

---

## Deferred: Big Five Population Study (Separate Paper)

**Adviser's bottom line**: Don't let population study excitement delay shipping improvements.

### If proceeding:
1. Pilot N=32 (8 clusters × 4), check R² at L22
2. If R² < 0.1, pivot approach
3. Ridge/ElasticNet, NOT OLS (VIF check mandatory)
4. Attachment style = most novel finding opportunity
5. Trajectory-based regression (Gemini), not just static snapshots
