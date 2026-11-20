# 7-minute talk script (7 slides)

About 850 words at a normal pace (≈130 words/min) is roughly 6.5 minutes, which leaves a little slack. The same text is in each slide's speaker notes in `final_presentation.pptx`.

---

## Slide 1: Sharper Pollution Maps You Can Trust (~30 s)
Hi, I'm Ryan, and this is Rency. Our project asks two questions: can computer vision turn a blurry satellite pollution map into a sharp one, and can the model tell us how much to trust each pixel? On the right is a real example from our 2023 test year: the 4-kilometer input the satellite gives, and our model's 1-kilometer prediction on a hazy day.

## Slide 2: Satellites see pollution too coarsely for a city (~60 s)
Satellites measure aerosol optical depth, or AOD, which tracks fine-particle pollution. Even NASA's sharpest product, MAIAC, is 1 kilometer per pixel, and many are several kilometers, too coarse to see a single highway. Our pipeline takes a 4-kilometer map and runs it through a super-resolution network. We compare three: EDSR, U-Net and SwinIR-light. The network outputs two maps: the sharp 1-kilometer AOD, and an uncertainty map, sigma-hat, that says how far off it expects to be at each pixel. We ask four questions. Is it more accurate than bicubic interpolation? Does it stay consistent with the coarse input? Does the uncertainty really know where the model is wrong? And does sharper AOD match ground monitors better?

## Slide 3: Six years of data, split by year so the test is fair (~55 s)
We used six years of daily MAIAC AOD over Dallas–Fort Worth. To keep the test honest we split by year: 2018 to 2021 for training, 2022 only to pick the best epoch and calibrate, and 2023 scored once at the very end. That's about 4,500 training patches, 1,100 for validation and almost 1,000 for testing. One clarification our professor asked about: bicubic is our baseline. It's a fixed formula with no training. It sees only the 4-kilometer input, and both bicubic and our models are graded against the same held-out 1-kilometer map. The true map is never an input to either; it's only the answer key.

## Slide 4: Deep SR beats bicubic, and SwinIR-light wins on every seed (~60 s)
Research question one: every model beats bicubic. This is test error on 2023, averaged over five training runs per model. Bicubic is at 12.6; our best model, SwinIR-light, is at 10.7. That's 15 percent lower, and it was the best on all five runs, so it's not a lucky run. U-Net and EDSR are tied. Research question two: we added a rule that forces the sharp map to average back exactly to the coarse input. The mismatch drops to essentially zero, about 10 to the minus 9, with no loss in accuracy and slightly more fine detail. So the model is not inventing pollution the satellite didn't see.

## Slide 5: The model knows where it is likely to be wrong (~65 s)
Research question three is the uncertainty. Each row is a 2023 test patch: a clear day, a cloudy day and a hazy day. From left: the input, our prediction, the truth, the model's uncertainty, and the actual error. The uncertainty map is made without ever seeing the truth, yet its dark spots line up with where the error is high. Across the whole year, pixels the model flags as uncertain have five to seven times more error. Even inside each 4-by-4 block it learned that corners are harder than centers, with a rank correlation of 0.99. The catch: the raw uncertainty was overconfident. SwinIR said 90 percent sure, but it was right only 79 percent of the time.

## Slide 6: A calibrated range that holds on a year it never saw (~70 s)
So we made the uncertainty trustworthy. First, a guaranteed range: we stretch sigma-hat by one factor, lambda-hat, fitted on 2022 with a statistical safety margin, so that at most 10 percent of pixels fall outside. On 2023 every model missed only about 6 to 8 percent, so the guarantee held. The plot shows why sigma-hat matters: a constant-width range with the same guarantee misses 17 to 20 percent at cloud edges and on hazy days. Using sigma-hat cuts that to 9 to 13 percent. Second, we found the cause of the overconfidence: training kept shrinking sigma after accuracy stopped improving. Picking the epoch by likelihood instead of error fixes most of it for free, from 79 to 87 percent, and an ensemble of five models is the best overall.

## Slide 7: What we found (~60 s)
To sum up: deep super-resolution sharpens AOD with 15 percent less error than bicubic, and the average-back rule keeps it faithful to what the satellite actually saw. The uncertainty map really points at the hard pixels, and after one calibration step it gives a range that held on a year the model never saw. The best recipe is to pick the epoch by likelihood, which is free, or use a five-model ensemble.

**[RENCY, ~20 s: RQ4, the EPA monitor comparison.]**

Our limits: the "truth" is MAIAC itself, the coarse input is simulated by averaging, and the guarantee assumes 2023 behaves like 2022. Thank you, we're happy to take questions.

---

## Likely questions (short answers)
- **Why does bicubic matter if you train against the true map?** Bicubic never sees the true map; it's the no-learning baseline. Both are graded on 2023, which neither method trained on.
- **Isn't σ̂ just the error?** No. σ̂ is predicted from the input alone, before the truth is seen. We then check it against the truth.
- **Why not just widen the range everywhere?** A constant range with the same guarantee misses 17–20% at cloud edges; σ̂ puts the width where the model is unsure.
- **Why is the guarantee approximate?** It assumes 2023 is like 2022, and patches from the same day are correlated.
