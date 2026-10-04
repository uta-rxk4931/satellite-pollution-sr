# Timeline, Sprints & Task Distribution

**Team:** Rency Ajit Kansagra, Ryan Seidel
**Start:** Oct 1, 2026
**Midterm proposal due:** Oct 20, 2026
**Final presentation:** Dec 1, 2026

## Overall shape

- **Oct 1 – Nov 4** (Sprints 1–5): Core project — RQ1 fidelity, RQ2 consistency, RQ3 uncertainty, first pass at RQ4 (EPA)
- **Nov 5 – Nov 18** (Sprints 6–7): Buffer — ablation table and figures. Stretch only if the core table is already done
- **Nov 19 – Dec 1** (Sprints 8–9): Write-up, slides, rehearsal

This matches the proposal timeline: midterm is RQ1 plus the written plan for consistency and uncertainty. After midterm we add the average-back constraint, then the confidence map and EPA \(R^2\), and use late November for the report.

## Research questions (from the proposal)

- **RQ1:** Can deep SR sharpen AOD maps (by \(s \geq 4\)) better than simple interpolation?
- **RQ2:** If we force the sharp map to average back to the coarse AOD, do we reduce made-up detail without just smoothing the map again?
- **RQ3:** Is the confidence map reliable? When it says 80% sure, is it correct about 80% of the time?
- **RQ4:** Does the sharpened AOD correlate more strongly with ground PM2.5 than the blurry version?

Required scale is \(s=4\) (4 km to 1 km). A road-density extra input, a smoke-day slice, or \(s=10\) are stretch goals, not required.

## Track split

Two tracks, with leadership **swapping after the midterm** so both members can explain the full pipeline on Dec 1.

- **Track A — Data & evaluation:** Earthdata / AppEEARS or GEE, patches, clouds, year split, EPA collocation, highway/monitor slices (RQ4)
- **Track B — Model & constraints:** SR backbone, average-back constraint (RQ2), uncertainty head, calibration (RQ3)

## Sprint-by-sprint plan

| Sprint | Dates | Rency | Ryan | Must-have by Thursday |
|---|---|---|---|---|
| 1 | Oct 1–7 | **Track A lead:** Earthdata login; pull a small DFW MAIAC slice (AppEEARS or GEE first, not the whole tile); confirm 0.001 scale and QA | **Track B lead:** pick a compact SR backbone, set up the training skeleton | One AOD map opens and looks like 0.05–0.4, not 50–400. Empty train script runs |
| 2 | Oct 8–14 | **Track A:** 64×64 patches, cloud/missing-pixel filter, \(s=4\) box-average degradation | **Track B:** baseline training end-to-end on dummy data, then a tiny real subset | Degradation function is frozen. One overfit-patch sanity check |
| 3 | Oct 15–21 (**midterm Oct 20**) | **Track A:** year split 2018–21 / 2022 / 2023; train-only log-transform or standardize | **Track B:** train on real \(s=4\) patches; first PSNR/SSIM/RMSE vs bicubic (RQ1) | Midterm text locked: four RQs, consistency plan, out of scope (no PDE, no PM2.5 retrieval). Both write it together |
| — | — | **swap leads** | **swap leads** | |
| 4 | Oct 22–28 | **Track B lead:** add the average-back constraint, \(\mathrm{AvgPool}_{s\times s}(\hat y) \approx x\), as a loss and/or a hard rescale (RQ2) | **Track A lead:** pull EPA AQS PM2.5 (88101) for DFW monitors; match to satellite overpass | Consistency error exists, with and without the constraint |
| 5 | Oct 29–Nov 4 | **Track B:** uncertainty head \((\hat y, \hat\sigma^2)\); first coverage / calibration plot (RQ3) | **Track A:** first RQ4 scatter — coarse AOD vs SR AOD vs EPA \(R^2\) | One figure or number for each of RQ1–RQ4, even if ugly |
| 6 | Nov 5–11 (buffer) | Ablation rows: bicubic / SR / SR+consistency / SR+consistency+\(\sigma\) | Same numbers into one table; freeze seed and splits | No new data source unless the table is already done |
| 7 | Nov 12–18 (buffer) | Side-by-side maps: clear day, cloudy day, high-AOD day | Optional stretch only if the table is frozen: road-density channel, smoke-day slice, or \(s=10\) | Figures and table frozen for the report |
| 8 | Nov 19–25 | Draft: Intro, Data, RQ2–RQ3 | Draft: Method, RQ1 and RQ4, limitations — build slides together | Shared slide outline |
| 9 | Nov 26–Dec 1 | Joint: rehearse, polish, dry run | Joint: rehearse, polish, dry run — **Dec 1 presentation** | Both can explain the loss and why a better PSNR that breaks the re-average check is a worse model |

## Sprint review cadence

- Weekly check-in every **Thursday**.
- Four questions only: what number moved (PSNR, consistency error, coverage, \(R^2\))? what is blocked? are we still inside scope? what is the single deliverable for next Thursday?

## Notes

- Sprints 6–7 are buffer. \(s=10\), a road-density channel, and a smoke-day slice are stretch goals from the Scope section, not required deliverables.
- Start with a DFW subset (AppEEARS or Google Earth Engine). Full `h09v05` tile download is optional if the laptop cannot hold ~20–30 GB of HDF.
- EPA monitors are evaluation only, never training.
- Both members should be able to explain the full pipeline by the final presentation, since track leadership rotates.
