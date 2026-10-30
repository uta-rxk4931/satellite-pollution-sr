# Reference papers

Papers directly relevant to our satellite image super-resolution (SR) project. Grouped by how they connect to our work.

## Core SR methods (baseline architecture ideas)
- Lanaras et al., "Super-resolution of Sentinel-2 images: Learning a globally applicable deep neural network" — https://ar5iv.labs.arxiv.org/html/1803.04271
- Chen et al., "Activating More Pixels in Image Super-Resolution Transformer" (CVPR 2023) — https://doi.org/10.1109/CVPR52729.2023.02142
- Wang, Chen, Hoi, "Deep Learning for Image Super-Resolution: A Survey" — https://doi.org/10.1038/s42254-021-00314-5
- Deudon et al., "HighRes-net: Recursive Fusion for Multi-Frame Super-Resolution of Satellite Imagery" — https://arxiv.org/abs/2002.06460

## Uncertainty-aware SR (Phase 1 add-on)
- Zhang et al., "Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model" (CVPR 2025) — https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Uncertainty-guided_Perturbation_for_Image_Super-Resolution_Diffusion_Model_CVPR_2025_paper.html
- "Diffusion-Powered Image Super-Resolution That You Can Actually Trust" (NeurIPS workshop) — https://neurips.cc/virtual/2024/105573
- "BUFF: Bayesian Uncertainty Guided Diffusion Probabilistic Model for Single Image Super-Resolution" — https://arxiv.org/pdf/2504.03490
- Zheng, Dewil, Arias, "Self-Supervised Uncertainty Estimation For Super-Resolution of Satellite Images" (2026) — https://arxiv.org/abs/2603.14074 (local copy: `uncertainity_pdfs/`). Shows the Gaussian NLL makes σ̂² the expected squared error per pixel; V-RMSE, coverage, sharpness, per-subgrid analysis. Basis for RQ3(a).
- Belhasin, Romano, Freedman, Rivlin, Elad, "Principal Uncertainty Quantification with Spatial Correlation for Image Restoration Problems" (PUQ, TPAMI 2024) — https://arxiv.org/abs/2305.10124 (local copy: `uncertainity_pdfs/`). Conformal risk control: calibrate interval scale so P(miss rate ≤ α) ≥ 1−δ. Basis for RQ3(b).
- Angelopoulos et al., "Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging" (im2im-uq, ICML 2022) — pixel-wise version of the same calibration, which is what RQ3(b) uses.

## Task-driven / downstream evaluation of SR (Phase 1 add-on)
- Shermeyer & Van Etten, "The Effects of Super-Resolution on Object Detection Performance in Satellite Imagery"

## Remote sensing SR (satellite-specific context)
- "A radiometrically and spatially consistent super-resolution framework for Sentinel-2 (SEN2SR)" — https://www.sciencedirect.com/science/article/pii/S0034425725006261
- "Deep learning-based Sentinel-2 super-resolution via channel attention and high-frequency feature enhancement" (2026) — https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1644460/full
- "Super-Resolution of Sentinel-2 Satellite Images: A Comparison of Different Interpolation Methods" (2026) — https://www.mdpi.com/2504-4990/8/1/14
- Digitalsense, "A Comprehensive Introduction to Super Resolution for Satellite Images" (blog) — https://www.digitalsense.ai/blog/introduction-to-super-resolution-for-satellite-images

## Physics-informed SR (Phase 2 — future work, not current scope)
- "Physics-Informed Super-Resolution of Atmospheric Data" (2026) — https://arxiv.org/html/2607.18877v1
- "Physics-Informed Computer Vision: A Review and Perspectives" — https://dl.acm.org/doi/10.1145/3689037
- Ren et al., "Physics-informed Deep Super-resolution for Spatiotemporal Data" — https://arxiv.org/abs/2208.01462

## Field context / podcast
- Satellite Image Deep Learning podcast (Robin Cole) — https://www.satellite-image-deep-learning.com/archive
