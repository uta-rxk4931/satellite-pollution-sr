"""
Builds final_paper/final_presentation.pptx: 7 slides (~1 min each) with the talk script in the speaker notes.
Run from the repo root:  python final_paper/make_slides.py <folder with hero_input/pred/sigma.png>
"""
import sys

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERO = sys.argv[1] if len(sys.argv) > 1 else "final_paper"
OUT = "final_paper/final_presentation.pptx"

# Palette: hazy-sky navy, haze amber, clean-air teal
NAVY, NAVY2 = RGBColor(0x1B, 0x24, 0x30), RGBColor(0x2A, 0x36, 0x47)
INK, MUTED = RGBColor(0x1B, 0x24, 0x30), RGBColor(0x5B, 0x65, 0x73)
AMBER, TEAL = RGBColor(0xE0, 0x9A, 0x2E), RGBColor(0x1F, 0x8A, 0x7E)
TINT, WHITE, SOFT = RGBColor(0xEE, 0xF2, 0xF5), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xC9, 0xD3, 0xDE)
GRAY = RGBColor(0xA3, 0xAA, 0xB3)
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None):
    """runs: a string, or a list of paragraphs; each paragraph is a string or a list of (text, {opts})."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for k, para in enumerate(paras):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = o.get("color", color)
    return box


def card(slide, x, y, w, h, fill=TINT, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.5)
    s.shadow.inherit = False
    return s


def dot(slide, x, y, d, label, fill=AMBER, color=WHITE, size=14):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), True, BODY, color


def arrow(slide, x, y, w=0.45, h=0.32, color=GRAY):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False


def background(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = color


def title(slide, t):
    text(slide, 0.6, 0.45, 12.1, 0.8, t, size=32, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)


def stat(slide, x, y, w, big, label, color=TEAL, big_size=40, gap=0.78):
    text(slide, x, y, w, 0.75, big, size=big_size, bold=True, font=HEAD, color=color)
    text(slide, x, y + gap, w, 0.8, label, size=15, color=MUTED)


def notes(slide, script):
    slide.notes_slide.notes_text_frame.text = script


# ---------------------------------------------------------------- 1. Title
s = prs.slides.add_slide(BLANK)
background(s, NAVY)
text(s, 0.75, 1.35, 6.6, 0.4, "CSE 6367 Computer Vision · Final presentation", size=14, color=AMBER, bold=True)
text(s, 0.75, 1.85, 6.6, 1.9, "Sharper Pollution Maps You Can Trust", size=44, bold=True, font=HEAD, color=WHITE)
text(s, 0.75, 3.85, 6.3, 1.0, "Uncertainty-aware super-resolution of satellite aerosol data over Dallas–Fort Worth",
     size=18, color=SOFT)
text(s, 0.75, 5.35, 6.3, 0.9, [[("Ryan Seidel  &  Rency Ajit Kansagra", {"bold": True, "color": WHITE})],
                                [("Instructor: Dr. Diego Patiño", {"color": SOFT, "size": 14})]], size=16)
s.shapes.add_picture(f"{HERO}/hero_input.png", Inches(7.55), Inches(2.2), Inches(2.45), Inches(2.45))
s.shapes.add_picture(f"{HERO}/hero_pred.png", Inches(10.4), Inches(2.2), Inches(2.45), Inches(2.45))
arrow(s, 10.0, 3.26, 0.36, 0.34, AMBER)
text(s, 7.55, 4.8, 2.45, 0.6, [[("4 km input", {"bold": True, "color": WHITE})],
                               [("what the satellite gives", {"size": 12, "color": SOFT})]], size=15,
     align=PP_ALIGN.CENTER)
text(s, 10.4, 4.8, 2.45, 0.6, [[("1 km prediction", {"bold": True, "color": WHITE})],
                                [("our model, hazy day in 2023", {"size": 12, "color": SOFT})]], size=15,
     align=PP_ALIGN.CENTER)
notes(s, "[~30 s] Hi, I'm Ryan, and this is Rency. Our project asks two questions: can computer vision turn a "
         "blurry satellite pollution map into a sharp one, and can the model tell us how much to trust each pixel? "
         "On the right is a real example from our 2023 test year: the 4-kilometer input the satellite gives, and "
         "our model's 1-kilometer prediction on a hazy day.")

# ---------------------------------------------------------------- 2. Problem + approach
s = prs.slides.add_slide(BLANK)
background(s, WHITE)
title(s, "Satellites see pollution too coarsely for a city")
steps = [("4 km satellite map", "aerosol optical depth (AOD)"),
         ("SR network", "EDSR · U-Net · SwinIR-light"),
         ("1 km map  ŷ\n+ uncertainty  σ̂", "how far off it expects to be"),
         ("Check vs. real 1 km map", "held-out 2023 MAIAC")]
x = 0.6
for k, (head, sub) in enumerate(steps):
    card(s, x, 1.6, 2.65, 1.55, fill=TINT)
    text(s, x + 0.2, 1.75, 2.25, 1.3, [[(head, {"bold": True, "size": 17})], [(sub, {"size": 13, "color": MUTED})]],
         anchor=MSO_ANCHOR.MIDDLE, spacing=4)
    if k < 3:
        arrow(s, x + 2.7, 2.2, 0.38, 0.34)
    x += 3.15
rqs = [("1", "Accuracy", "Sharper and more accurate than bicubic?"),
       ("2", "Consistency", "Does the sharp map average back to the input?"),
       ("3", "Uncertainty", "Does σ̂ know where the model is wrong?"),
       ("4", "Ground truth", "Does it match EPA PM2.5 monitors better?")]
x = 0.6
for n, head, sub in rqs:
    dot(s, x, 3.75, 0.55, f"RQ{n}", fill=TEAL if n != "3" else AMBER, size=12)
    text(s, x + 0.7, 3.72, 2.25, 1.6, [[(head, {"bold": True, "size": 17})], [(sub, {"size": 14, "color": MUTED})]],
         spacing=3)
    x += 3.15
text(s, 0.6, 5.4, 12.1, 0.9,
     "Even NASA's sharpest aerosol product (MAIAC) is 1 km per pixel; many are several km, too coarse to see one "
     "highway. Ground monitors are accurate but sparse.", size=15, color=MUTED)
notes(s, "[~60 s] Satellites measure aerosol optical depth, or AOD, which tracks fine-particle pollution. Even NASA's "
         "sharpest product, MAIAC, is 1 kilometer per pixel, and many are several kilometers, too coarse to see a "
         "single highway. Our pipeline takes a 4-kilometer map and runs it through a super-resolution network. We "
         "compare three: EDSR, U-Net and SwinIR-light. The network outputs two maps: the sharp 1-kilometer AOD, and "
         "an uncertainty map, sigma-hat, that says how far off it expects to be at each pixel. We ask four questions. "
         "Is it more accurate than bicubic interpolation? Does it stay consistent with the coarse input? Does the "
         "uncertainty really know where the model is wrong? And does sharper AOD match ground monitors better?")

# ---------------------------------------------------------------- 3. Data + fair test
s = prs.slides.add_slide(BLANK)
background(s, WHITE)
title(s, "Six years of data, split by year so the test is fair")
years = [("2018", TEAL), ("2019", TEAL), ("2020", TEAL), ("2021", TEAL), ("2022", AMBER), ("2023", NAVY)]
for k, (yr, col) in enumerate(years):
    card(s, 0.6 + k * 2.03, 1.6, 1.88, 0.75, fill=col)
    text(s, 0.6 + k * 2.03, 1.6, 1.88, 0.75, yr, size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
stat(s, 0.6, 2.65, 7.9, "4,543 patches", "train (2018–2021): the model learns from these", TEAL)
stat(s, 8.72, 2.65, 2.0, "1,142", "2022: pick epoch + calibrate", AMBER)
stat(s, 10.75, 2.65, 2.1, "976", "2023: scored once, at the end", NAVY)
card(s, 0.6, 4.65, 12.1, 2.05, fill=TINT)
dot(s, 0.9, 4.95, 0.5, "?", fill=AMBER, size=16)
text(s, 1.65, 4.9, 10.8, 1.7,
     [[("What does bicubic do? ", {"bold": True}), ("It is the no-learning baseline: a fixed formula that sees only "
       "the same 4 km input as our network and never sees the true map.", {})],
      [("The real 1 km map is only the answer key: ", {"bold": True}),
       ("it makes the 4 km input, is the training target (2018–21), and grades both methods (2023).", {})]],
     size=16, spacing=8)
notes(s, "[~55 s] We used six years of daily MAIAC AOD over Dallas-Fort Worth. To keep the test honest we split by "
         "year: 2018 to 2021 for training, 2022 only to pick the best epoch and calibrate, and 2023 scored once at the "
         "very end. That's about 4,500 training patches, 1,100 for validation and almost 1,000 for testing. One "
         "clarification our professor asked about: bicubic is our baseline. It's a fixed formula with no training. "
         "It sees only the 4-kilometer input, and both bicubic and our models are graded against the same held-out "
         "1-kilometer map. The true map is never an input to either; it's only the answer key.")

# ---------------------------------------------------------------- 4. RQ1 + RQ2
s = prs.slides.add_slide(BLANK)
background(s, WHITE)
title(s, "Deep SR beats bicubic, and SwinIR-light wins on every seed")
cd = CategoryChartData()
cd.categories = ["Bicubic", "U-Net", "EDSR", "SwinIR-light"]
cd.add_series("RMSE (×10⁻³)", (12.6, 11.0, 11.0, 10.7))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.55), Inches(6.9), Inches(5.0), cd)
ch = gf.chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "2023 test error, RMSE ×10⁻³ (mean of 5 runs)"
tp = ch.chart_title.text_frame.paragraphs[0]
tp.runs[0].font.size, tp.runs[0].font.bold, tp.runs[0].font.name = Pt(14), False, BODY
tp.runs[0].font.color.rgb = MUTED
ch.font.size, ch.font.name, ch.font.color.rgb = Pt(14), BODY, INK
plot = ch.plots[0]
plot.gap_width = 70
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = "0.0", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold = Pt(16), True
for k, col in enumerate([GRAY, TEAL, TEAL, AMBER]):
    pt = plot.series[0].points[k]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 0.0, 14.0  # start at 0 so bar heights are honest
va.has_major_gridlines = False
va.visible = False
ch.category_axis.format.line.color.rgb = SOFT
ch.category_axis.tick_labels.font.size = Pt(14)
card(s, 7.95, 1.6, 4.75, 2.35, fill=TINT)
text(s, 8.25, 1.75, 4.2, 0.4, "RQ1 · ACCURACY", size=12, bold=True, color=TEAL)
stat(s, 8.25, 2.1, 4.2, "−15% error", "vs. bicubic for SwinIR-light, the best model on all 5 seeds", TEAL)
card(s, 7.95, 4.2, 4.75, 2.35, fill=TINT)
text(s, 8.25, 4.35, 4.2, 0.4, "RQ2 · CONSISTENCY", size=12, bold=True, color=TEAL)
stat(s, 8.25, 4.7, 4.2, "≈ 0 mismatch", "averages back to the input (8.7×10⁻⁹) with the same accuracy and more detail",
     TEAL)
notes(s, "[~60 s] Research question one: every model beats bicubic. This is test error on 2023, averaged over five "
         "training runs per model. Bicubic is at 12.6; our best model, SwinIR-light, is at 10.7. That's 15 percent "
         "lower, and it was the best on all five runs, so it's not a lucky run. U-Net and EDSR are tied. Research "
         "question two: we added a rule that forces the sharp map to average back exactly to the coarse input. The "
         "mismatch drops to essentially zero, about 10 to the minus 9, with no loss in accuracy and slightly more "
         "fine detail. So the model is not inventing pollution the satellite didn't see.")

# ---------------------------------------------------------------- 5. RQ3(a)
s = prs.slides.add_slide(BLANK)
background(s, WHITE)
title(s, "The model knows where it is likely to be wrong")
s.shapes.add_picture("final_paper/fig_examples_swinir.png", Inches(0.6), Inches(1.5), Inches(7.6), Inches(4.78))
text(s, 0.6, 6.4, 7.6, 0.6, "2023 test patches (SwinIR-light). σ̂ is made without seeing the truth; cyan = outside "
                            "the calibrated range.", size=12, color=MUTED)
stat(s, 8.75, 1.6, 4.0, "5–7×", "more real error on the pixels the model flags as most uncertain", AMBER, 48)
stat(s, 8.75, 3.25, 4.0, "0.99", "rank correlation: it learned that block corners are harder than centers", AMBER,
     48)
card(s, 8.75, 4.95, 3.95, 1.45, fill=TINT)
text(s, 9.0, 5.07, 3.5, 1.25, [[("But raw σ̂ was overconfident: ", {"bold": True}),
                                ("SwinIR said “90% sure” and was right 79% of the time.", {})]], size=15,
     anchor=MSO_ANCHOR.MIDDLE)
notes(s, "[~65 s] Research question three is the uncertainty. Each row is a 2023 test patch: a clear day, a cloudy "
         "day and a hazy day. From left: the input, our prediction, the truth, the model's uncertainty, and the "
         "actual error. The uncertainty map is made without ever seeing the truth, yet its dark spots line up with "
         "where the error is high. Across the whole year, pixels the model flags as uncertain have five to seven "
         "times more error. Even inside each 4-by-4 block it learned that corners are harder than centers, with a "
         "rank correlation of 0.99. The catch: the raw uncertainty was overconfident. SwinIR said 90 percent sure, "
         "but it was right only 79 percent of the time.")

# ---------------------------------------------------------------- 6. RQ3(b) + fix
s = prs.slides.add_slide(BLANK)
background(s, WHITE)
title(s, "A calibrated range that holds on a year it never saw")
s.shapes.add_picture("final_paper/fig_rq3b_range.png", Inches(0.6), Inches(1.55), Inches(7.6), Inches(3.69))
card(s, 0.6, 5.5, 7.6, 1.2, fill=TINT)
text(s, 0.85, 5.6, 7.1, 1.0, [[("Range = ŷ ± λ̂·σ̂.  ", {"bold": True}),
                              ("λ̂ is fitted on 2022 with a safety margin (conformal risk control) so at most 10% of "
                               "pixels fall outside.", {})]], size=15, anchor=MSO_ANCHOR.MIDDLE)
stat(s, 8.75, 1.6, 4.0, "6.4–7.7%", "of 2023 pixels outside the range, for every model (target ≤ 10%)", TEAL, 44)
stat(s, 8.75, 3.2, 4.0, "17–20% → 9–13%", "misses at cloud edges and hazy days: constant range vs. σ̂ range", TEAL,
     30, gap=0.6)
card(s, 8.75, 4.75, 3.95, 1.95, fill=TINT)
text(s, 9.0, 4.88, 3.5, 1.75, [[("Fixing overconfidence", {"bold": True, "color": AMBER})],
                               [("Pick epoch by likelihood: 79% → 87% (free)", {})],
                               [("5-model ensemble: best overall (RMSE 0.0104, 91% coverage)", {})]], size=14,
     spacing=4)
notes(s, "[~70 s] So we made the uncertainty trustworthy. First, a guaranteed range: we stretch sigma-hat by one "
         "factor, lambda-hat, fitted on 2022 with a statistical safety margin, so that at most 10 percent of pixels "
         "fall outside. On 2023 every model missed only about 6 to 8 percent, so the guarantee held. The plot shows "
         "why sigma-hat matters: a constant-width range with the same guarantee misses 17 to 20 percent at cloud "
         "edges and on hazy days. Using sigma-hat cuts that to 9 to 13 percent. Second, we found the cause of the "
         "overconfidence: training kept shrinking sigma after accuracy stopped improving. Picking the epoch by "
         "likelihood instead of error fixes most of it for free, from 79 to 87 percent, and an ensemble of five "
         "models is the best overall.")

# ---------------------------------------------------------------- 7. Conclusion
s = prs.slides.add_slide(BLANK)
background(s, NAVY)
text(s, 0.6, 0.45, 12.1, 0.8, "What we found", size=36, bold=True, font=HEAD, color=WHITE,
     anchor=MSO_ANCHOR.MIDDLE)
takeaways = [("1", "Sharper and faithful", "15% less error than bicubic; the sharp map averages back to the "
                                           "satellite input exactly."),
             ("2", "Uncertainty you can trust", "σ̂ points at the hard pixels; one calibration step gives a range "
                                                "that held on 2023."),
             ("3", "Best recipe", "Pick the epoch by likelihood (free); a 5-model ensemble is the most accurate.")]
for k, (n, head, sub) in enumerate(takeaways):
    x = 0.6 + k * 4.1
    card(s, x, 1.6, 3.85, 2.75, fill=NAVY2)
    dot(s, x + 0.3, 1.85, 0.55, n, fill=AMBER, size=16)
    text(s, x + 0.3, 2.6, 3.3, 1.7, [[(head, {"bold": True, "size": 19, "color": WHITE})],
                                     [(sub, {"size": 15, "color": SOFT})]], spacing=6)
card(s, 0.6, 4.65, 12.1, 1.05, fill=NAVY, line=AMBER)
text(s, 0.9, 4.72, 11.5, 0.9, [[("RQ4 · EPA PM2.5 monitors  ", {"bold": True, "color": AMBER}),
                                ("Rency: does sharper AOD match ground monitors better?", {"color": WHITE})]],
     size=17, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 5.95, 8.6, 0.9, "Limits: the “truth” is MAIAC itself, the 4 km input is simulated by averaging, and "
                             "the guarantee assumes 2023 behaves like 2022.", size=14, color=SOFT)
text(s, 9.4, 5.95, 3.3, 0.9, "Thank you! Questions?", size=22, bold=True, font=HEAD, color=WHITE,
     align=PP_ALIGN.RIGHT)
notes(s, "[~60 s] To sum up: deep super-resolution sharpens AOD with 15 percent less error than bicubic, and the "
         "average-back rule keeps it faithful to what the satellite actually saw. The uncertainty map really points "
         "at the hard pixels, and after one calibration step it gives a range that held on a year the model never "
         "saw. The best recipe is to pick the epoch by likelihood, which is free, or use a five-model ensemble. "
         "[RENCY, ~20 s: RQ4, the EPA monitor comparison.] Our limits: the 'truth' is MAIAC itself, the coarse input "
         "is simulated by averaging, and the guarantee assumes 2023 behaves like 2022. Thank you, we're happy to "
         "take questions.")

prs.save(OUT)
print("saved", OUT)
