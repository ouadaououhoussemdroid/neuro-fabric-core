# Neuro-Fabric Core Roadmap: Next Phase

## Executive Summary

This document outlines the immediate next phase of development for the Neuro-Fabric Core repository, based on the comprehensive technical audit, maturity assessment, and technical debt analysis. The roadmap focuses on transitioning the platform from a **Research Platform** (63/100 maturity) to an **MVP-Ready** state (70+ maturity) within a 4-6 month timeframe.

The approach prioritizes **risk reduction** and **foundational hardening** before advancing core functionality, ensuring that each improvement builds upon a stable, secure, and observable base.

## Guiding Principles

1. **Risk-First Approach:** Address critical operational risks (CI/CD, security, reliability) before feature development
2. **Incremental Delivery:** Deliver value in small, verifiable increments with feedback loops
3. **Maintain Backward Compatibility:** Ensure changes don't break existing functionality unless absolutely necessary
4. **Observability First:** Instrument changes to enable measurement and data-driven decisions
5. **Technical Debt Reduction:** Allocate dedicated effort to remediate high-impact debt items
6. **Validation Gate:** Establish empirical validation checkpoints before scaling claims

## Phase Overview: Foundation Hardening & Core Enablement (Months 1-4)

### Phase 1: Operational Foundation (Weeks 1-4)

**Goal:** Establish production-grade operational baselines for security, reliability, and velocity.

| Week | Objective                            | Key Activities                                                                                                                                                                                                   | Success Criteria                                                                               |
| ---- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 1    | **Implement CI/CD Pipeline**         | - Configure GitHub Actions workflow<br>- Add `bunx vitest run` and `bunx typecheck` on PR<br>- Set up coverage reporting (target: >80%)<br>- Configure branch protection rules                                   | All PRs blocked on failing tests; main branch protected; visibility into test coverage         |
| 1-2  | **Enhance Security Posture**         | - Add rate limiting (100 req/min/user) to `/api/eeg/upload`<br>- Implement upload size limit (100MB)<br>- Add filename sanitization and path traversal protection                                                | No DoS vulnerability via upload endpoint; malicious files rejected; safe filename handling     |
| 2-3  | **Achieve WASM Independence**        | - Complete self-hosted ORT WASM bundle in `/ort/`<br>- Implement SHA-384 integrity verification via `manifest.json`<br>- Establish fallback hierarchy: self-hosted → jsDelivr → PCA                              | WASM loads from self-hosted by default; CDN fallback functional; integrity verification active |
| 3-4  | **Establish Observability Baseline** | - Add Prometheus metrics endpoint (`/metrics`)<br>- Instrument key latency EEGs (upload → embedding → storage)<br>- Implement fallback rate alerting (>5% triggers warning)<br>- Add basic health check endpoint | Metrics endpoint exposed; alerts configured; baseline performance data collected               |

### Phase 2: Core Capability Enhancement (Weeks 5-8)

**Goal:** Improve core EEG processing, persistence, and cognitive capabilities while maintaining operational excellence.

| Week | Objective                               | Key Activities                                                                                                                                                                                                                                                    | Success Criteria                                                                                                                        |
| ---- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 5    | **Optimize Signal Processing**          | - Replace O(M²) DFT with FFT in `src/lib/embeddings/features.ts`<br>- Implement as configurable feature flag (fft-enabled)<br>- Add benchmark comparison showing 2-5x speedup<br>- Verify embedding parity (<0.001 cosine difference)                             | 2-5x speedup in feature extraction; no regression in embedding quality; feature flag for A/B testing                                    |
| 5-6  | **Implement Persistent Vector Storage** | - Create pgvector migration script<br>- Establish cosine ANN index with model-ID tagging<br>- Backfill existing in-memory vectors to Postgres<br>- Implement dual-write during transition                                                                         | Vector search persists across reloads; cross-device retrieval functional; recall@10 maintained; fallback to in-memory during transition |
| 6-7  | **Develop Cognitive Decoder**           | - Train logistic regression/SVM on public dataset (e.g., EEG eyes-open/closed)<br>- Export to ONNX with attention/workload/arousal outputs (3-D)<br>- Register as `cognitive-decoder-v1`<br>- Implement feature flag to enable/disable vs heuristic               | Decoder model loads and predicts; heuristic behind feature flag; A/B test framework ready; initial validation complete                  |
| 7-8  | **Integrate Real EEG Dataset Loader**   | - Implement Sleep-EDF loader in `src/lib/datasets/`<br>- Add BCI-IV-2a loader via MOABB (reference existing training pipeline)<br>- Create dataset abstraction layer with format detection<br>- Add basic metadata extraction (duration, channels, sampling rate) | Users can load and process Sleep-EDF files; metadata extracted correctly; foundation for future dataset expansion                       |

### Phase 3: Platform Maturation & Validation (Months 3-4)

**Goal:** Prepare for internal pilot and establish validation gates for future development.

| Month | Objective                             | Key Activities                                                                                                                                                                                                            | Success Criteria                                                                                            |
| ----- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 3     | **Implement Model Operations**        | - Implement content-hashed model storage (e.g., Supabase Storage)<br>- Add SHA verification at load time using manifest<br>- Create model versioning metadata tracking<br>- Enable model rollback capability              | Models served from content-hashed URLs; tamper detection; ability to rollback to previous versions          |
| 3-4   | **Enhance Observability & Debugging** | - Add distributed tracing (OpenTelemetry)<br>- Implement endpoint-specific SLO dashboards<br>- Create operational runbook for common issues<br>- Add detailed logging for model loading and fallback events               | Trace ID propagation; SLO dashboards visible; runbook reduces MTTR; improved diagnosability                 |
| 3-4   | **Establish Validation Gates**        | - Define BCI-IV-2a holdout evaluation protocol<br>- Create automated validation script<br>- Establish recall@10 threshold for model promotion (e.g., ≥0.65 vs PCA ~0.40)<br>- Document validation results in model claims | Repeatable validation process; objective criteria for model readiness; transparent reporting of performance |

### Phase 4: Pilot Preparation & Execution (Months 4-6)

**Goal:** Validate with real users and prepare for external release.

| Month | Objective                            | Key Activities                                                                                                                                                                                                           | Success Criteria                                                                                     |
| ----- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| 4     | **Internal Pilot Readiness**         | - Deploy to internal staging environment<br>- Onboard 3-5 internal users (researchers, engineers)<br>- Create user onboarding flow and documentation<br>- Establish feedback collection mechanism                        | System stable under load; users able to complete core workflows; NPS >30 from internal users         |
| 5     | **External Validation**              | - Conduct BCI-IV-2a holdout evaluation<br>- Measure recall@10 vs PCA baseline<br>- Perform statistical significance testing (p<0.05)<br>- Publish technical validation report                                            | Recall@10 ≥0.65 (vs PCA ~0.40); statistical significance achieved; validation documentation complete |
| 5-6   | **Feedback Integration & Hardening** | - Prioritize and implement user-reported issues<br>- Address discovered bugs and edge cases<br>- Prepare public-facing documentation and tutorials<br>- Finalize security and data handling documentation for compliance | Issue backlog reduced by 50%; public-facing docs complete; readiness for limited external beta       |

## Success Metrics for MVP Readiness

To determine when the platform achieves MVP readiness, we will track the following metrics:

### Operational Excellence

| Metric                    | Target        | Measurement Method                             |
| ------------------------- | ------------- | ---------------------------------------------- |
| **CI Success Rate**       | >95%          | Percentage of PRs passing CI on first attempt  |
| **Deployment Frequency**  | ≥2/week       | Count of successful deployments to staging     |
| **Change Failure Rate**   | <5%           | Percentage of deployments causing incidents    |
| **Mean Time to Recovery** | <30 min       | Average time to restore service after incident |
| **System Uptime**         | 99.5% monthly | Uptime excluding planned maintenance           |

### Security & Compliance

| Metric                       | Target          | Measurement Method                                            |
| ---------------------------- | --------------- | ------------------------------------------------------------- |
| **Critical Vulnerabilities** | 0               | Results from automated security scans (OWASP ZAP, Snyk, etc.) |
| **Rate Limit Effectiveness** | 100% block rate | Percentage of abusive requests blocked during testing         |
| **Upload Abuse Prevention**  | 100% block rate | Percentage of oversized/malicious uploads blocked             |
| **Data Encryption**          | AES-256 at rest | Verification of encryption for sensitive data fields          |

### Performance & Reliability

| Metric            | Target  | Measurement Method                                    |
| ----------------- | ------- | ----------------------------------------------------- |
| **P50 Latency**   | <500ms  | End-to-end: upload → embedding → storage confirmation |
| **P95 Latency**   | <1000ms | 95th percentile of end-to-end latency                 |
| **Fallback Rate** | <1%     | Percentage of embedding requests falling back to PCA  |
| **Error Rate**    | <0.1%   | Percentage of requests returning 5xx errors           |

### Functional Completeness

| Metric                         | Target            | Measurement Method                                                 |
| ------------------------------ | ----------------- | ------------------------------------------------------------------ |
| **Core Workflow Success**      | >95%              | Percentage of users completing upload → process → store → retrieve |
| **Cognitive Decoder Adoption** | >50% opt-in       | Percentage of users enabling the cognitive decoder feature         |
| **Dataset Utilization**        | >3 types          | Number of distinct EEG formats successfully processed              |
| **Model Validation**           | Published results | Publicly available validation report meeting scientific standards  |

## Risk Management

### Technical Risks & Mitigation Strategies

| Risk                              | Probability | Impact | Mitigation                                                                   |
| --------------------------------- | ----------- | ------ | ---------------------------------------------------------------------------- |
| **WASM Compatibility Issues**     | Medium      | High   | Maintain jsDelivr fallback; test across Chrome, Firefox, Safari, Edge        |
| **pgvector Migration Complexity** | Medium      | High   | Implement dual-write strategy; extensive staging testing; rollback procedure |
| **Cognitive Decoder Performance** | Medium      | Medium | Start with simple models; use transfer learning; set realistic expectations  |
| **Validation Failure**            | Low         | High   | Set achievable targets; use established baselines; iterate on preprocessing  |
| **Scope Creep**                   | High        | Medium | Strict change control; defer non-MVP features to post-MVP backlog            |

### Contingency Plans

- If WASM self-hosting proves problematic: Extend jsDelivr contract with SLA; implement stricter CSP
- If pgvector migration delays: Optimize in-memory index with LRU eviction; plan migration as post-MVP effort
- If cognitive decoder underperforms: Extend heuristic with individual baselines; position as enhancement rather than replacement
- If validation targets not met: Publish results transparently; focus on other value propositions (usability, security, features)

## Resource Allocation

### Team Structure (Recommended)

- **1 Full-Stack Engineer** (Backend/Infrastructure focus) - 50% allocation
- **1 Full-Stack Engineer** (Frontend/ML focus) - 50% allocation
- **0.5 FTE DevOps/Infrastructure** - Shared responsibility for CI/CD, monitoring, infrastructure
- **0.5 FTE Data Scientist/ML Engineer** - Focus on cognitive decoder training and validation
- **0.25 FTE Product/User Research** - Focus on feedback collection and usability testing

### Effort Distribution by Phase

| Phase                | Infrastructure (40%) | Core Features (35%) | Observability (15%) | Validation (10%) |
| -------------------- | -------------------- | ------------------- | ------------------- | ---------------- |
| Phase 1 (Weeks 1-4)  | 70%                  | 20%                 | 10%                 | 0%               |
| Phase 2 (Weeks 5-8)  | 30%                  | 50%                 | 15%                 | 5%               |
| Phase 3 (Months 3-4) | 20%                  | 40%                 | 20%                 | 20%              |
| Phase 4 (Months 4-6) | 10%                  | 30%                 | 20%                 | 40%              |

## Milestone-Based Progress Tracking

### Milestone 1: Operational Foundation (End of Week 4)

- [ ] CI/CD pipeline protecting main branch
- [ ] Rate limiting and upload validation active
- [ ] Self-hosted WASM with integrity verification
- [ ] Basic observability (metrics, health checks)

### Milestone 2: Core Capabilities (End of Week 8)

- [ ] FFT-optimized feature extraction
- [ ] Persistent vector storage with pgvector
- [ ] Trained cognitive decoder model available
- [ ] Sleep-EDF loader functional

### Milestone 3: Platform Maturation (End of Month 4)

- [ ] Content-hashed model storage with verification
- [ ] Enhanced observability (tracing, SLOs, runbook)
- [ ] Validation gate established and documented
- [ ] Internal pilot successfully onboarded

### Milestone 4: MVP Readiness (End of Month 6)

- [ ] Published BCI-IV-2a validation report
- [ ] All success metrics met or exceeded
- [ ] Security and compliance documentation complete
- [ ] Ready for limited external beta or internal production use

## Decision Gates

### Gate 1: Proceed to Core Features (After Week 4)

**Go/No-Go Criteria:**

- CI/CD pipeline blocking merges with failing tests
- Rate limiting and upload validation in place and tested
- WASM self-hosting functional with fallback
- Basic metrics endpoint operational

### Gate 2: Proceed to Platform Maturation (After Week 8)

**Go/No-Go Criteria:**

- Feature-flagged FFT implementation validated
- Persistent storage working with basic query performance
- Cognitive decoder model loads and produces reasonable outputs
- Sleep-EDF loader able to read and process files

### Gate 3: Proceed to Pilot (After Month 4)

**Go/No-Go Criteria:**

- Model versioning and verification functional
- Observability suite providing actionable insights
- Validation gate producing repeatable results
- Internal users able to complete core workflows

### Gate 4: Approve for Limited External Release (After Month 6)

**Go/No-Go Criteria:**

- Validation report shows statistically significant improvement over PCA
- All success metrics meet targets
- Security and privacy documentation reviewed
- Feedback incorporation process established

## Conclusion

This roadmap provides a clear, risk-aware path from the current research platform state to MVP readiness. By front-loading operational excellence and systematically addressing the most critical technical debt items, we create a stable foundation upon which to build advanced features.

The phased approach ensures that each increment delivers measurable value while reducing overall project risk. Regular decision gates provide opportunities to course-correct based on empirical evidence rather than assumptions.

With disciplined execution of this plan, the Neuro-Fabric Core platform can achieve MVP readiness within 4-6 months, positioning it for initial user pilots, feedback collection, and eventual preparation for broader release or commercial exploration.
