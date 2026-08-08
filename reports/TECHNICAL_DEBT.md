# Neuro-Fabric Core Technical Debt Register

## Executive Summary

This document catalogs all known technical debt items in the Neuro-Fabric Core repository, prioritized by impact and effort. Technical debt refers to implied costs of additional rework caused by choosing an easy (limited) or suboptimal solution now instead of using a better approach that would take longer.

Each debt item is classified by:

- **Impact:** Potential negative consequences if not addressed (Low, Medium, High, Critical)
- **Effort:** Estimated work required to resolve (Low: 1-3 days, Medium: 3-5 days, High: 1-2 weeks, Very High: 3+ weeks)
- **Priority:** Combined assessment for remediation ordering (P1: Critical/Immediate, P2: High/Soon, P3: Medium/Future)

Addressing this technical debt is essential for transitioning from a research platform to a production-ready MVP.

## Technical Debt Inventory

### Critical Priority (P1) - Must Fix Before MVP Release

| ID     | Component      | Description                                                                                                                                                                                | Impact                                                                                                                           | Effort                                                                                                     | Priority |
| ------ | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------- |
| TD-001 | Infrastructure | **No CI/CD Pipeline** - Tests and type checks are not gated on pull requests, enabling regressions to slip into main branch undetected.                                                    | **Critical** - High risk of undetected breaks in core functionality, especially dangerous for AI model changes                   | **Low** - 1-2 days to implement GitHub Actions workflow with `bunx vitest run` and typecheck               | **P1**   |
| TD-002 | Security       | **Missing Rate Limiting & Upload Validation** - `/api/eeg/upload` endpoint lacks request rate limits and file size validation, enabling denial-of-service attacks and resource exhaustion. | **Critical** - Service availability risk; malicious users could crash service or exhaust resources                               | **Low** - 1-2 days to implement middleware with token bucket algorithm and size validation                 | **P1**   |
| TD-003 | Reliability    | **WASM CDN Single Point of Failure** - Reliance on jsDelivr CDN for ONNX Runtime Web WASM with no guaranteed SLA or integrity verification.                                                | **Critical** - CDN outage causes silent fallback to PCA for all users worldwide; potential CSP violations in strict environments | **Medium** - 3-5 days to complete self-hosted ORT bundle with SHA-384 verification and fallback hierarchy  | **P1**   |
| TD-004 | ML Operations  | **Model Artifact Bundling** - ONNX model bundled in `public/` directory increases cold load size and lacks content-hashed URLs for cache busting and integrity verification.               | **High** - Inefficient caching; no tamper detection; complicates updates and rollbacks                                           | **Low** - 2-3 days to implement content-housed storage (e.g., Supabase Storage) with manifest verification | **P2**   |

### High Priority (P2) - Should Fix for MVP Release

| ID     | Component           | Description                                                                                                                                                                                          | Impact                                                                                                                  | Effort                                                                                                                                         | Priority |
| ------ | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| TD-005 | Performance         | **Inefficient Feature Extraction** - O(M²) DFT implementation in `src/lib/embeddings/features.ts` instead of O(M log M) FFT creates unnecessary latency ceiling at longer window lengths.            | **Medium** - 2-5x slower feature extraction for 4s windows; limits scalability to longer epochs                         | **Low** - 1 day to replace DFT with FFT implementation, maintaining API compatibility                                                          | **P2**   |
| TD-006 | Persistence         | **In-Memory Vector Index** - `NeuralVectorIndex` loses all state on page reload, preventing cross-device retrieval and persistent collections.                                                       | **Medium** - Poor user experience; inability to build persistent knowledge bases or longitudinal studies                | **Medium** - 3-5 days to implement pgvector migration with cosine ANN index and model-ID tagging                                               | **P2**   |
| TD-007 | ML Functionality    | **Heuristic Cognitive Decoder** - Attention/workload/arousal computed from simple band ratios (e.g., alpha/beta) rather than trained models, limiting scientific utility and individual calibration. | **Medium** - Reduced predictive accuracy; no personalization; limited to population-level heuristics                    | **Medium** - 3-5 days to train and deploy lightweight model (logistic regression → shallow NN) on public dataset with labeled cognitive states | **P2**   |
| TD-008 | Observability       | **Lack of Metrics & Alerting** - No Prometheus endpoint, distributed tracing, or SLO-based alerting makes production monitoring and troubleshooting difficult.                                       | **Medium** - Inability to measure SLAs, detect performance degradation, or proactively address issues                   | **Medium** - 3-5 days to add metrics endpoint, instrument key paths, and configure basic alerting                                              | **P2**   |
| TD-009 | Dataset Integration | **Missing Real EEG Loaders** - No integrated loaders for major EEG datasets (Sleep-EDF, CHB-MIT, TUH), limiting research utility and external validation.                                            | **Medium** - Increases friction for researchers wanting to validate on standard datasets; requires custom preprocessing | **Medium** - 3-5 days to implement Sleep-EDF loader as starting point, using MNE-Python or similar                                             | **P2**   |
| TD-010 | Security Hardening  | **Insufficient Input Validation** - Filename sanitization missing on upload endpoints; potential path traversal if files ever stored on filesystem.                                                  | **Low-Medium** - Theoretical risk if storage implementation changes; low immediate impact with current Supabase storage | **Low** - 1 day to implement filename sanitization and path traversal protection                                                               | **P2**   |

### Medium Priority (P3) - Nice to Fix Post-MVP

| ID     | Component            | Description                                                                                                                                                                | Impact                                                                                      | Effort                                                                                                       | Priority |
| ------ | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------- |
| TD-011 | Code Modularity      | **Acquisition Source Coupling** - Tight coupling between acquisition sources (BrainFlow, LSL) and hardware-specific APIs makes adding new sources cumbersome.              | **Low-Medium** - Slows integration of new hardware modalities; increases maintenance burden | **Medium** - 3-5 days to define `AcquisitionSource` interface and refactor existing implementations          | **P3**   |
| TD-012 | Scalability          | **Linear Vector Search** - In-memory implementation uses linear scan rather than approximate nearest neighbor (ANN) algorithms, limiting scalability to large collections. | **Low-Medium** - Only affects users with >10k vectors; acceptable for initial MVP scale     | **Medium** - 3-5 days to implement IVF-PQ or HNSW via pgvector extension post-migration                      | **P3**   |
| TD-013 | Documentation        | **Outdated/Historical Documents** - Some documents marked as "Historical" risk confusion if not read carefully; lack of clear deprecation notices.                         | **Low** - Minor confusion potential; low impact on functionality                            | **Low** - 1-2 days to archive or clearly label outdated documents, add deprecation banners where needed      | **P3**   |
| TD-014 | Test Coverage        | **Inconsistent Test Coverage** - Some modules well-tested (adapters, validation), others minimally tested (EEG processing, acquisition), creating maintenance risks.       | **Low** - Increases risk of bugs in less-tested areas; makes refactoring more dangerous     | **Medium** - 3-5 days to increase coverage in low-test areas to 80%+                                         | **P3**   |
| TD-015 | Developer Experience | **Missing Contributing Guidelines** - No `CONTRIBUTING.md` or developer onboarding documentation increases friction for external contributors.                             | **Low** - Slows community growth; increases burden on maintainers for basic questions       | **Low** - 1 day to create basic contributing guide with setup instructions, coding standards, and PR process | **P3**   |

## Debt Burden Analysis

### By Priority:

- **P1 (Critical):** 4 items - Must address before MVP release
- **P2 (High):** 6 items - Should address for MVP release
- **P3 (Medium):** 5 items - Nice to address post-MVP

### By Impact:

- **Critical:** 2 items (TD-001, TD-002)
- **High:** 2 items (TD-003, TD-004)
- **Medium:** 10 items (TD-005 through TD-014)
- **Low:** 3 items (TD-011, TD-013, TD-015)

### By Effort:

- **Low:** 9 items (TD-001, TD-002, TD-004, TD-005, TD-010, TD-013, TD-015)
- **Medium:** 9 items (TD-003, TD-006, TD-007, TD-008, TD-009, TD-011, TT-012, TD-014)
- **High:** 0 items
- **Very High:** 0 items

### Estimated Total Effort:

- **P1 Items:** 7-14 days
- **P2 Items:** 18-30 days
- **P3 Items:** 15-25 days
- **Total Addressable Debt:** 40-69 days (approximately 2-3 months of focused effort)

## Debt Reduction Strategy

### Recommended Approach: Debt Snowball with Critical Path Focus

**Phase 1: Immediate Stabilization (Weeks 1-2)**
Focus on P1 items to establish a stable, secure foundation:

1. Implement CI pipeline (TD-001) - 2 days
2. Add rate limiting and upload validation (TD-002) - 2 days
3. Begin WASM self-hosting implementation (TD-003) - start work

**Phase 2: Foundation Hardening (Weeks 3-4)**
Continue critical path and begin high-impact items:

1. Complete WASM self-hosting with fallback hierarchy (TD-003) - 3 days
2. Implement content-hashed model storage (TD-004) - 3 days
3. Replace DFT with FFT (TD-005) - 1 day
4. Begin pgvector migration (TD-006) - start work

**Phase 3: Core Functionality (Weeks 5-6)**
Address remaining high-priority items:

1. Complete pgvector migration with ANN index (TD-006) - 3 days
2. Train and deploy cognitive decoder (TD-007) - 4 days
3. Add metrics endpoint and alerting (TD-008) - 3 days
4. Implement Sleep-EDF loader (TD-009) - 3 days

**Phase 4: Polish and Extend (Weeks 7-8)**
Address medium-priority items and prepare for MVP:

1. Add filename sanitization (TD-010) - 1 day
2. Refactor acquisition sources (TD-011) - 4 days
3. Enhance test coverage (TD-014) - 3 days
4. Create contributing guidelines (TD-015) - 1 day
5. Archive outdated documentation (TD-013) - 1 day

## Risk Mitigation for Debt Reduction

### Technical Risks:

- **Risk:** Changing core algorithms (e.g., DFT→FFT) could affect model compatibility
  - **Mitigation:** Implement as feature flag; validate embedding parity before and after change
- **Risk:** Migrating to pgvector could introduce latency or consistency issues
  - **Mitigation:** Implement dual-write during transition; validate performance parity
- **Risk:** Self-hosting WASM could introduce browser compatibility issues
  - **Mitigation:** Maintain CDN fallback; test across major browsers

### Schedule Risks:

- **Risk:** Underestimating effort for complex items (e.g., pgvector migration)
  - **Mitigation:** Time-box spikes; prototype before full implementation
- **Risk:** Dependencies on external factors (e.g., labeled data for cognitive decoder)
  - **Mitigation:** Scope to available datasets; use transfer learning or synthetic labels if needed

### Quality Risks:

- **Risk:** Rushing debt reduction introduces new bugs
  - **Mitigation:** Maintain test coverage; use feature flags for risky changes; employ canary releases

## Progress Tracking

### Definition of Done for Debt Items:

1. **Code Change:** Implement solution in codebase
2. **Testing:** Add/update tests to cover new functionality
3. **Documentation:** Update relevant documentation (API, architecture, user guides)
4. **Verification:** Manual testing in staging environment
5. **Review:** Code review approval by at least one maintainer
6. **Monitoring:** Post-deployment observation for regressions

### Metrics for Success:

- **Defect Escape Rate:** Reduction in bugs found in production post-release
- **Deployment Frequency:** Increase in safe deployment cadence
- **Mean Time to Recovery (MTTR):** Reduction due to better observability
- **Change Failure Rate:** Reduction in failed deployments
- **System Availability:** Increase in uptime percentage

## Consequences of Non-Action

If technical debt remains unaddressed:

1. **TD-001 (No CI/CD):** Increased regression risk; slower development velocity due to manual testing
2. **TD-002 (No Rate Limiting):** Service vulnerability to abuse; potential unexpected costs from resource exhaustion
3. **TD-003 (WASM CDN SPOF):** Global service degradation during CDN outages; compliance issues in regulated environments
4. **TD-004 (Model Bundling):** Poor user experience for returning users; complicated model update process
5. **TD-005 (Inefficient Features):** Unnecessary latency; poor scalability to longer EEG epochs
6. **TD-006 (In-Memory Vectors):** Poor user experience; inability to build persistent knowledge bases
7. **TD-007 (Heuristic Decoder):** Limited clinical/research utility; inability to individualize assessments
8. **TD-008 (No Observability):** Blind spots in production; difficulty meeting SLA requirements
9. **TD-009 (Missing Loaders):** Reduced appeal to researchers; validation friction
10. **TD-010 (Input Validation):** Theoretical security risk; potential future vulnerabilities

## Investment Justification

Addressing this technical debt represents a high-leverage investment:

- **Risk Reduction:** Eliminates known failure modes that could damage reputation or cause data loss
- **Velocity Improvement:** Reduces unplanned work and increases predictable delivery capacity
- **Quality Enhancement:** Improves reliability, performance, and security posture
- **Market Readiness:** Positions product for paid user acquisition and potential regulatory pathways
- **Maintainability:** Reduces long-term ownership costs and simplifies future enhancements

The estimated 2-3 months of focused effort to address P1 and P2 items represents a sound investment given the potential consequences of inaction and the accelerated path to market readiness it provides.
