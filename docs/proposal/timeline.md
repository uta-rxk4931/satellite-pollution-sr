# Timeline, Sprints & Task Distribution

**Team:** Rency Ajit Kansagra, Ryan Seidel
**Start:** Oct 1, 2026
**Midterm proposal due:** Oct 20, 2026
**Final presentation:** Dec 1, 2026

## Overall shape

- **Oct 1 – Nov 4** (Sprints 1–5): Core project build — RQ1, RQ2, RQ3 working end-to-end
- **Nov 5 – Nov 18** (Sprints 6–7): Buffer — stretch goals (s=10, seasonal analysis, model improvements), polish results
- **Nov 19 – Dec 1** (Sprints 8–9): Write-up, slides, rehearsal

## Track split

Two tracks, with leadership **swapping after the midterm** so both members get hands-on experience with both the data-engineering side and the model/uncertainty side of the project.

- **Track A — Data & Evaluation:** data access, preprocessing, patching, EPA correlation (RQ3)
- **Track B — Model & Uncertainty:** SR architecture, training loop, confidence output, calibration (RQ2)

## Sprint-by-sprint plan

| Sprint | Dates | Rency | Ryan |
|---|---|---|---|
| 1 | Oct 1–7 | **Track A lead:** NASA Earthdata access, download MAIAC tile h09v05, explore raw data | **Track B lead:** pick SR architecture, set up training skeleton |
| 2 | Oct 8–14 | **Track A:** build patch extraction (64×64), cloud/missing-pixel filtering, s=4 degradation function | **Track B:** get baseline model training end-to-end on dummy data |
| 3 | Oct 15–21 (**Midterm due Oct 20**) | **Track A:** finalize train/val/test split by year (2018–21 / 2022 / 2023), log-transform/standardize | **Track B:** train on real s=4 patches, first PSNR/SSIM vs. bicubic — **both write midterm proposal together** |
| — | — | **↓ swap leads ↓** | **↓ swap leads ↓** |
| 4 | Oct 22–28 | **Track B lead:** add uncertainty/confidence output, set up RQ2 calibration testing | **Track A lead:** pull EPA AQS PM2.5 data for DFW monitors, prep RQ3 comparison |
| 5 | Oct 29–Nov 4 | **Track B:** finish RQ2 calibration check ("90% confident = right 90% of the time?") | **Track A:** run RQ3 correlation (sharpened AOD vs. EPA monitors, R²) |
| 6 | Nov 5–11 (buffer) | Stretch: try s=10, or seasonal/smoke-day error analysis | Stretch: try improving model (attention layer, second architecture) |
| 7 | Nov 12–18 (buffer) | Qualitative side-by-side comparison figures | Finalize metrics tables, comparisons vs. baselines (CNN/transformer refs) |
| 8 | Nov 19–25 | Draft report: Intro, Data, RQ2 results | Draft report: Method, Model, RQ1/RQ3 results — **build slides together** |
| 9 | Nov 26–Dec 1 | Joint: rehearse, polish, dry run | Joint: rehearse, polish, dry run — **Dec 1 presentation** |

## Sprint review cadence

- Weekly check-in every **Thursday** — brief update on what got done, blockers, and next sprint's plan.

## Notes

- Sprints 6–7 are intentionally buffer/stretch — s=10 and additional model improvements are explicitly stretch goals, not required deliverables (see Scope section of proposal).
- Both members are expected to be able to explain the full pipeline (not just their current track) by the final presentation, since track leadership rotates.
