"""
Script to synchronize IEEE_PAPER.html with PAPER/paper_final.tex,
embedding all figures as Base64 Data URIs so they display with 100% reliability in Edge PDF rendering.
"""

import base64
from pathlib import Path
import subprocess

REPO_DIR = Path(__file__).resolve().parent
TEX_PATH = REPO_DIR / "PAPER" / "paper_final.tex"
HTML_PATH = REPO_DIR / "IEEE_PAPER.html"
PDF_PATH = REPO_DIR / "PAPER" / "paper_final.pdf"
ROOT_PDF = REPO_DIR / "IEEE_PAPER.pdf"
FIG_DIR = REPO_DIR / "reports" / "figures"

def get_base64_img(img_name: str) -> str:
    path = FIG_DIR / img_name
    if not path.exists():
        path = REPO_DIR / "PAPER" / "figures" / img_name
    if not path.exists():
        print(f"Warning: Figure {img_name} not found!")
        return ""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def main():
    fig1_b64 = get_base64_img("fig1_pipeline_architecture.png")
    fig2_b64 = get_base64_img("fig2_personalized_baseline.png")
    fig3_b64 = get_base64_img("Figure_4_GMM_BIC_AIC.png")
    fig4_b64 = get_base64_img("fig3_four_experiments_diagram.png")
    fig5_b64 = get_base64_img("fig4_silhouette_experiments.png")
    fig6_b64 = get_base64_img("fig_domain_shift_cohen.png")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cross-Domain Generalization of Personalized Wearable Health State Discovery: A Synthetic-to-Real Evaluation</title>
<style>
@page {{
    size: letter;
    margin: 0.75in 0.625in 0.75in 0.625in;
}}
body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 9.5pt;
    line-height: 1.15;
    color: #000;
    background-color: #fff;
    margin: 0;
    padding: 0;
}}

.paper-page {{
    width: 8.5in;
    height: 11in;
    padding: 0.58in 0.65in 0.55in 0.65in;
    margin: 0 auto 10px auto;
    background: #fff;
    box-sizing: border-box;
    position: relative;
    page-break-after: always;
    page-break-inside: avoid;
}}

.columns {{
    column-count: 2;
    column-gap: 0.22in;
    height: 100%;
}}

.paper-title {{
    font-size: 20pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 12pt;
    margin-top: 4pt;
    line-height: 1.25;
}}

.author-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-around;
    text-align: center;
    font-size: 9.5pt;
    margin-bottom: 12pt;
    line-height: 1.25;
}}

.author-box {{
    width: 46%;
    margin-bottom: 8pt;
    text-align: center;
}}

.author-box-center {{
    width: 46%;
    margin: 0 auto 8pt auto;
    text-align: center;
}}

.abstract-box {{
    width: 100%;
    margin: 0 0 10pt 0;
    font-size: 9pt;
    font-weight: bold;
    text-align: justify;
    line-height: 1.38;
    font-family: "Times New Roman", Times, serif;
}}
.abstract-heading {{
    font-weight: bold;
    font-style: italic;
}}
.keywords-heading {{
    font-weight: bold;
    font-style: italic;
    margin-top: 5pt;
}}
h2.section-title {{
    font-size: 9.5pt;
    font-weight: bold;
    text-transform: uppercase;
    margin-top: 5pt;
    margin-bottom: 2pt;
    text-align: center;
    break-after: avoid;
}}
h3.subsection-title {{
    font-size: 9.5pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 3.5pt;
    margin-bottom: 1.5pt;
    break-after: avoid;
}}
p {{
    text-indent: 1.2em;
    margin: 0 0 4pt 0;
    line-height: 1.38;
    text-align: justify;
}}
p.first-p {{
    text-indent: 0;
}}
.table-box {{
    margin: 6pt 0;
    break-inside: avoid;
    width: 100%;
}}
.paper-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 4pt 0;
    font-size: 8.5pt;
    line-height: 1.3;
    text-align: center;
    border: 1.5pt solid #000;
}}
.paper-table th {{
    border: 1px solid #000;
    background-color: #f1f5f9;
    padding: 3pt 4pt;
    font-weight: bold;
    color: #000;
}}
.paper-table td {{
    border: 1px solid #000;
    padding: 2.5pt 3pt;
}}
.table-caption {{
    font-size: 8.5pt;
    text-transform: uppercase;
    text-align: center;
    font-weight: bold;
    margin-bottom: 3pt;
    line-height: 1.25;
}}
.figure-box {{
    text-align: center;
    margin: 4pt 0;
    break-inside: avoid;
    width: 100%;
}}
.figure-box img {{
    width: 100%;
    max-width: 100%;
    max-height: 1.8in;
    height: auto;
    display: block;
    margin: 0 auto;
    border: 0.5pt solid #cbd5e1;
    border-radius: 2px;
    object-fit: contain;
}}
.figure-caption {{
    font-size: 8.5pt;
    text-align: justify;
    line-height: 1.32;
    margin-top: 4pt;
}}
ol, ul {{
    padding-left: 1.3em;
    margin: 5pt 0;
    line-height: 1.42;
}}
li {{
    margin-bottom: 3pt;
    line-height: 1.42;
}}
.references {{
    font-size: 8.5pt;
    line-height: 1.35;
    text-align: justify;
}}
.ref-item {{
    text-indent: -1.5em;
    padding-left: 1.5em;
    margin-bottom: 3.5pt;
    line-height: 1.35;
    break-inside: avoid;
    page-break-inside: avoid;
}}
.cite-link {{
    color: #0056b3;
    text-decoration: none;
    font-weight: normal;
}}
.cite-link:hover {{
    text-decoration: underline;
}}
</style>
</head>
<body>

<div class="paper-title">Cross-Domain Generalization of Personalized Wearable Health<br>State Discovery: A Synthetic-to-Real Evaluation</div>

<div class="author-grid">
    <div class="author-box">
        <strong>1<sup>st</sup> Nigam L. Raj</strong><br>
        <em>Dept. of Information Science and Engineering</em><br>
        <em>Nitte Meenakshi Institute of Technology</em><br>
        Bengaluru, India<br>
        nigamraj711@gmail.com
    </div>
    <div class="author-box">
        <strong>2<sup>nd</sup> Prajwal R</strong><br>
        <em>Dept. of Information Science and Engineering</em><br>
        <em>Nitte Meenakshi Institute of Technology</em><br>
        Bengaluru, India<br>
        prajwalgowda81234@gmail.com
    </div>
    <div class="author-box">
        <strong>3<sup>rd</sup> S. Lokesh</strong><br>
        <em>Dept. of Information Science and Engineering</em><br>
        <em>Nitte Meenakshi Institute of Technology</em><br>
        Bengaluru, India<br>
        mail2lokesh022@gmail.com
    </div>
    <div class="author-box">
        <strong>4<sup>th</sup> Sachin S. K</strong><br>
        <em>Dept. of Information Science and Engineering</em><br>
        <em>Nitte Meenakshi Institute of Technology</em><br>
        Bengaluru, India<br>
        sachinsk711@gmail.com
    </div>
    <div class="author-box-center">
        <strong>5<sup>th</sup> Lakshmi M</strong><br>
        <em>Dept. of Information Science and Engineering</em><br>
        <em>Nitte Meenakshi Institute of Technology</em><br>
        Bengaluru, India<br>
        lakshmi.m@nmit.ac.in
    </div>
</div>

<div class="columns">

<div class="abstract-box">
    <span class="abstract-heading">Abstract—</span>Personalized health-state monitoring from consumer wearables typically relies on population thresholds or supervised models requiring costly ground-truth labels. We present an unsupervised, longitudinal framework that discovers latent physiological states (<em>Recovery, Baseline, Strain</em>) via a Gaussian Mixture Model (GMM) over per-user causal deviation features, with a Hidden Markov Model (HMM) capturing temporal transitions. A key contribution is systematic cross-domain evaluation: we test whether a state-discovery model trained entirely on synthetic data generalizes to real-world Fitbit data, and whether synthetic augmentation improves discovery when real data is limited. Under leakage-free, user-disjoint 5-fold cross-validation, zero-shot synthetic-to-real transfer achieves a silhouette score of 0.2117 ± 0.0098, closely matching the real-trained baseline (0.2168 ± 0.0078), with a 2.3% relative gap that is not statistically significant after Nadeau–Bengio variance correction and Holm–Bonferroni adjustment (p_adj = 0.1139). Synthetic data augmentation yields negligible benefit when moderate amounts of real-world training data are available (61 users); all tested augmentation ratios produce small performance decreases relative to real-only training. Non-parametric statistical alignment against Ecological Momentary Assessment (EMA) self-report labels (N = 1,751) reveals an exploratory uncorrected trend in self-reported happiness (HAPPY, H = 7.0991, p = 0.0287 uncorrected) with Recovery showing highest positive affect (0.30 ± 0.46), but this association does not survive Holm–Bonferroni multiple-comparison adjustment (p_adj = 0.2012). We report these results with full statistical rigor including permutation-null baselines, explicit model-selection audits, and a normalization caveat analysis, providing an honest characterization of consistent physiological structure that transfers across synthetic and real domains.
    <div class="keywords-heading">Index Terms—wearable computing, unsupervised learning, Gaussian mixture model, hidden Markov model, domain adaptation, synthetic data, health informatics, cross-domain transfer</div>
</div>

<h2 class="section-title">I. Introduction</h2>
<p class="first-p">Consumer wearable devices continuously record heart rate, sleep, and activity signals, but converting these raw streams into meaningful health intelligence remains difficult. Two dominant approaches exist. The first compares readings against fixed population thresholds (e.g., "resting heart rate above 100 bpm is high"), which ignores substantial inter-individual variation in normal physiology <a href="#ref-6" class="cite-link">[6]</a>, <a href="#ref-8" class="cite-link">[8]</a>. The second trains supervised classifiers against clinical or self-reported labels, which are expensive to collect at scale and rarely available in longitudinal consumer datasets <a href="#ref-3" class="cite-link">[3]</a>, <a href="#ref-4" class="cite-link">[4]</a>.</p>
<p>An alternative, growing in the wearable-AI literature, is unsupervised discovery of latent physiological states relative to each individual's own recent history <a href="#ref-1" class="cite-link">[1]</a>, <a href="#ref-5" class="cite-link">[5]</a>. This avoids the labeling bottleneck entirely, at the cost of requiring careful validation: without ground-truth labels, it is not obvious whether discovered "states" reflect genuine physiological structure or artifacts of the clustering procedure.</p>
<p>A second, largely unaddressed question in this space is <em>data availability</em>. Real longitudinal wearable cohorts with sufficient per-user history are small and hard to obtain, while synthetic physiological data generators can produce arbitrarily large cohorts cheaply. It is unclear whether a state-discovery model developed entirely on synthetic data transfers to real users, and whether synthetic data can substitute for, or augment, scarce real training data. This question echoes the sim-to-real transfer literature well established in robotics and vision, but has seen comparatively little systematic treatment for longitudinal physiological time series in unsupervised settings.</p>
<p>This paper makes three contributions. First, we present a personalized, longitudinal state-discovery framework combining causal 7-day rolling baselines, a harmonized composite severity score, GMM-based latent state discovery, and HMM-based temporal modeling. Second, we design a leakage-free, user-disjoint cross-domain evaluation protocol comparing four conditions: synthetic-only training, real-only training, zero-shot synthetic-to-real transfer, and synthetic-augmented real training at controlled mixing ratios. Third, we report this evaluation with a level of statistical rigor uncommon in small-cohort wearable studies: repeated user-level cross-validation with the Nadeau–Bengio variance correction <a href="#ref-19" class="cite-link">[19]</a>, Holm–Bonferroni correction for multiple comparisons <a href="#ref-20" class="cite-link">[20]</a>, permutation-null baselines, and an explicit audit of whether our reported domain-shift reduction is a genuine finding or an artifact of per-user normalization.</p>

<div class="figure-box">
    <img src="{fig1_b64}" alt="Architecture Diagram" style="max-height: 1.5in; width: auto; max-width: 100%; object-fit: contain;">
    <div class="figure-caption"><strong>Fig. 1.</strong> End-to-end leakage-controlled pipeline architecture. All hyperparameter and scaler fitting occurs exclusively on training-fold data.</div>
</div>

<p>The empirical finding is honest: clustering structure is weak by conventional standards <a href="#ref-21" class="cite-link">[21]</a> across every condition tested, yet remarkably <em>consistent</em> across synthetic, real, and transfer conditions. We emphasize consistency over absolute magnitude, and we report this transparency to support reproducibility and to enable future work to build on rather than replicate mistakes.</p>

<h2 class="section-title">II. Related Work</h2>
<p class="first-p"><strong>Wearable health inference.</strong> Rashid et al. combine multimodal sensor fusion with context awareness for stress detection <a href="#ref-1" class="cite-link">[1]</a>. Saad et al. and Sabry et al. survey machine learning approaches for wearable healthcare, noting the scarcity of labeled longitudinal data as a persistent bottleneck <a href="#ref-3" class="cite-link">[3]</a>, <a href="#ref-4" class="cite-link">[4]</a>. Dunn et al. and Li et al. demonstrate that individualized, longitudinal physiological monitoring detects anomalies invisible to population-level thresholds <a href="#ref-5" class="cite-link">[5]</a>, <a href="#ref-8" class="cite-link">[8]</a>, motivating our personalized-baseline design. Seshadri et al. and Rogers et al. review the hardware and continuous-monitoring landscape that produces this class of data <a href="#ref-6" class="cite-link">[6]</a>, <a href="#ref-10" class="cite-link">[10]</a>. Topol discusses the broader convergence of wearable sensing and AI in clinical contexts <a href="#ref-7" class="cite-link">[7]</a>.</p>
<p><strong>Federated and transfer learning for wearables.</strong> Chen et al. propose FedHealth, a federated transfer-learning framework that adapts a source model to heterogeneous wearable users without centralizing data <a href="#ref-2" class="cite-link">[2]</a>. Our zero-shot and augmentation experiments address a related but distinct question: not how to adapt a model to new users via federated updates, but whether a model trained purely on <em>synthetic</em> data transfers to <em>real</em> users without any target-domain fine-tuning at all, and whether mixing synthetic and real data during training helps.</p>
<p><strong>Reproducibility and evaluation rigor.</strong> Kapoor and Narayanan document how data leakage inflates reported performance across machine-learning-based science, particularly in small-cohort biomedical settings <a href="#ref-9" class="cite-link">[9]</a>. We adopt their concerns directly: every experiment in this work uses user-disjoint cross-validation, train-only scaler and hyperparameter fitting, and corrected statistical testing to avoid exactly this failure mode.</p>
<p><strong>Model foundations.</strong> Our latent-state model builds on standard Gaussian mixture modeling <a href="#ref-16" class="cite-link">[16]</a> and hidden Markov modeling <a href="#ref-14" class="cite-link">[14]</a>, with Bayesian Information Criterion model selection <a href="#ref-18" class="cite-link">[18]</a>. Factorial HMM extensions <a href="#ref-13" class="cite-link">[13]</a> are a natural direction for future multi-modality work. We use scikit-learn <a href="#ref-11" class="cite-link">[11]</a> for GMM fitting and evaluation, and note SHAP-based explainability <a href="#ref-12" class="cite-link">[12]</a> as a complementary direction we do not pursue in this paper. The real-world dataset used in this work is LifeSnaps <a href="#ref-15" class="cite-link">[15]</a>, a publicly released, peer-reviewed, multi-modal longitudinal dataset of Fitbit-derived physiological and behavioral signals.</p>

<h2 class="section-title">III. Datasets and Preprocessing</h2>
<h3 class="subsection-title">A. Datasets</h3>
<p class="first-p">We use two longitudinal wearable datasets. The <em>synthetic</em> dataset simulates 300 users over 184 days (55,200 daily observations). Per-user baseline means were sampled from Gaussian population distributions: resting heart rate ~ N(64.5, 8.0) bpm, daytime average heart rate ~ N(87.1, 10.0) bpm, daily steps ~ N(9282, 2200) count, distance ~ N(7.43, 1.8) km, calories ~ N(2074, 350) kcal, and sleep duration ~ N(6.99, 1.1) hours. Daily observations incorporate continuous temporal dynamics via an AR(1) process (phi = 0.65) with weekend activity shifts (±12% step/sleep variation) and 5% uniform random missingness. Crucially, the synthetic generator does <em>not</em> hardcode discrete <em>Recovery/Baseline/Strain</em> state labels or explicit cluster centroids; latent structure emerges naturally from continuous physiological variance. The <em>real</em> dataset is drawn from LifeSnaps <a href="#ref-15" class="cite-link">[15]</a>, a publicly released, multi-modal, longitudinal dataset collected from n = 71 participants across two study rounds (May–July 2021 and November 2021–January 2022), containing Fitbit Sense-derived physiological and behavioral signals alongside survey and ecological momentary assessment data. From the full multi-modal release we extract 7,410 daily Fitbit-derived records across the 71 de-identified participants.</p>

<div class="figure-box">
    <img src="{fig2_b64}" alt="Personalized Baseline" style="max-height: 1.5in; width: auto; max-width: 100%; object-fit: contain;">
    <div class="figure-caption"><strong>Fig. 2.</strong> Personalized causal baseline calculation showing raw resting heart rate, 7-day causal rolling mean, and extracted relative z-deviation (z_{{i,t}}).</div>
</div>

<p><strong>Provenance and ethics.</strong> LifeSnaps is a peer-reviewed dataset with a dedicated data-descriptor publication <a href="#ref-15" class="cite-link">[15]</a>, released under a CC-BY 4.0 license via Zenodo. Participant consent and study ethics procedures are documented by the original dataset authors, and all participants are de-identified in the released data. We report results as a non-clinical academic research use consistent with the dataset's intended purpose, and make no individual clinical claims.</p>

<h3 class="subsection-title">B. Authenticity-First Preprocessing</h3>
<p class="first-p">Real wearable data exhibits substantial sensor-specific missingness (SpO2: 82.9%, HRV: 66.6%, sleep: 52.1%, resting HR: 40.3%, steps: 35.5%). Rather than mass-imputing these gaps, we adopt an authenticity-first policy: we retain only a <em>common feature space</em> of six signals present with tractable missingness in both datasets — resting heart rate, average daytime heart rate, steps, distance, calories, and sleep duration — explicitly excluding HRV and SpO2 from the primary analysis rather than fabricating them. Within this common space, missing values are filled only via causal, within-user forward-fill with a maximum gap of two days; no population-mean or cohort-mean imputation is applied. This yields a primary real dataset of 4,159 observations across 69 users (56.1% row retention, 97.2% user retention), of which 84.1% of cells are fully observed and 15.9% are short causal fills. Zero cells in this primary set derive from any mean-imputation method; we verified this directly by tracing all 4,159 rows back to their raw source records.</p>
<p>An HMM-eligibility gate further restricts the subset used for temporal (HMM) evaluation: a user timeline qualifies only if it has at least 14 total observations and at least 7 post-warmup valid observations, ensuring one full calibration window plus a minimally usable evaluation sequence. This retains 61 of 69 users (4,115 rows), and this exact eligible set is reused, unchanged, across every real-data experiment to avoid comparing HMM results over shifting populations.</p>

<h2 class="section-title">IV. Methodology</h2>
<h3 class="subsection-title">A. System Architecture</h3>
<p class="first-p">The system architecture maps raw wearable streams into the six-feature common space, passes features through causal 7-day rolling baseline calculation, converts relative z-deviations, and fits GMM and HMM sequence decoders strictly on training-fold data.</p>

<h3 class="subsection-title">B. Personalized Baselines and Severity</h3>
<p class="first-p">For user i, feature f, and day t, the causal 7-day rolling baseline mean and standard deviation are computed over preceding days. Per-feature z-deviations are computed as z_{{i,t,f}} = (x_{{i,t,f}} - mu_{{i,t,f}}) / (sigma_{{i,t,f}} + eps). A composite severity score aggregates the six z-deviations into a single scalar: S_{{i,t}} = (1/6) sum(|z_{{i,t,f}}|), capturing overall deviation magnitude regardless of direction. This score serves as the ordering criterion for GMM component labels (Recovery = lowest, Baseline = middle, Strain = highest) and enables consistent cross-domain comparisons even when independently fit models produce different discrete state indices.</p>

<h3 class="subsection-title">C. Latent State Discovery</h3>
<p class="first-p">We fit a Gaussian Mixture Model with K = 3 diagonal-covariance components over the six-dimensional deviation vector. Components are deterministically ordered by median S_{{i,t}} and labeled Recovery (lowest), Baseline (middle), and Strain (highest); these are heuristic data-driven proxy labels applied to unsupervised clusters, not clinical diagnoses. Model-selection sweeps using the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC) across candidate cluster counts K in {{2, 3, 4, 5}} identify a minimum at K = 3 (BIC drops to 142,100 at K = 3 compared to 147,500 at K = 2, before rising to 143,800 at K = 4 and 145,900 at K = 5; Fig. 3). This quantitative result supports K = 3 as the preferred candidate under the evaluated range while matching domain interpretability into macro-physiological states.</p>

<div class="figure-box">
    <img src="{fig3_b64}" alt="BIC Model Selection" style="max-height: 1.5in; width: auto; max-width: 100%; object-fit: contain;">
    <div class="figure-caption"><strong>Fig. 3.</strong> Model selection for optimal cluster count. BIC and AIC criteria exhibit a minimum among evaluated candidate values K in {{2,3,4,5}} at K = 3 (BIC = 142,100), supporting K = 3 as the preferred candidate for macro-physiological state discovery.</div>
</div>

<p>We separately verified this is not a degenerate full-covariance artifact: a full-covariance K = 3 fit achieves a numerically lower BIC (47,125 vs. 67,900) but collapses cluster quality (silhouette 0.167 vs. 0.217, Davies–Bouldin 3.02 vs. 1.54), traceable to a near-singular component covariance (minimum eigenvalue 3.09e-4). Note that for this covariance-structure audit, BIC values (67,900 diagonal vs. 47,125 full) are evaluated directly on the primary single real cohort (N = 3,749 post-warmup rows) and are reported on a different observation scale than the 5-fold aggregated evaluation folds (N ~ 9,000 rows, BIC = 142,100 at K = 3). We use diagonal covariance throughout to avoid covariance collapse.</p>

<h3 class="subsection-title">D. Temporal Modeling</h3>
<p class="first-p">A discrete-emission Hidden Markov Model with three states is fit over the GMM-assigned daily state sequence per user timeline. The GMM state labels serve as categorical observations to a three-state HMM, whose transition and emission probabilities are estimated from the training sequences; Viterbi decoding <a href="#ref-14" class="cite-link">[14]</a> then provides the temporally smoothed latent-state sequence. While empirical frequency tables simply count static label transitions, the Categorical HMM models underlying sequence dynamics, Viterbi path decoding, and multi-day state persistence under a stationary Markov assumption. State 2 (Strain) exhibits the highest diagonal transition probability (0.76), reflecting observed multi-day temporal clustering rather than acute single-day fluctuations.</p>

<h3 class="subsection-title">E. Cross-Domain Experimental Design</h3>
<div class="figure-box">
    <img src="{fig4_b64}" alt="Four Experiments Diagram" style="max-height: 1.5in; width: auto; max-width: 100%; object-fit: contain;">
    <div class="figure-caption"><strong>Fig. 4.</strong> Four-experiment cross-domain evaluation framework architecture across synthetic and real-world cohorts.</div>
</div>
<p class="first-p">We define four experiments under a common 5-fold, user-level, user-disjoint cross-validation protocol (train and test user sets are strictly disjoint in every fold; scalers and K are fit on training-fold data only):</p>
<ul>
    <li><strong>Exp. 1 (Synthetic -> Synthetic):</strong> train and evaluate on synthetic users, as a controlled baseline.</li>
    <li><strong>Exp. 2 (Real -> Real):</strong> train and evaluate on real users, using the HMM-eligible 61-user set.</li>
    <li><strong>Exp. 3 (Synthetic -> Real, zero-shot):</strong> train exclusively on synthetic users; evaluate, without any refitting, on the identical real held-out test users used in Exp. 2. For zero-shot transfer, the scaler fitted on the synthetic training fold was applied unchanged to the real test fold; no real-domain statistics were used.</li>
    <li><strong>Exp. 4 (Synthetic+Real -> Real):</strong> train on real training users combined with synthetic users at controlled ratios (1:1, 2:1, 4:1 synthetic:real), evaluated on the same real test users. K is frozen from Exp. 1/2 and reused unchanged across all ratios; synthetic and real training sets are standard-scaled on their respective domain-specific training statistics prior to merging to align feature scale ranges so ratio mixing evaluates structural component overlap rather than scale disparity.</li>
</ul>

<h3 class="subsection-title">E. Baseline Models</h3>
<p class="first-p">To test whether latent state discovery captures genuine multivariate structure beyond the composite severity score S = (1/6) sum(|z_i,t,f|), we implement a deterministic <em>Severity-Tercile</em> baseline. Quantile thresholds (q33, q66) are computed exclusively on training-fold user severity distributions (D_train) and applied unchanged to held-out test users (D_eval). For zero-shot transfer (Exp. 3), synthetic training thresholds are applied directly to real test users with zero real set leakage. For secondary comparison, a hard-assignment K-Means (K = 3) baseline is fit on training-fold scaled deviations and evaluated on test-fold features, sorted by median severity score.</p>

<h3 class="subsection-title">F. Statistical Testing</h3>
<p class="first-p">For each pairwise comparison, we compute fold-paired mean differences and apply the Nadeau–Bengio corrected resampled t-test <a href="#ref-19" class="cite-link">[19]</a>, followed by Holm–Bonferroni correction <a href="#ref-20" class="cite-link">[20]</a> across comparison families. We report paired-sample effect sizes (d_z) across cross-validation folds. Permutation-null floors (50 shuffles per domain) confirm that clustering exceeds chance-level noise, and size-matched Calinski–Harabasz scores (N = 823) control for test fold size differences.</p>

<h2 class="section-title">V. Results</h2>
<h3 class="subsection-title">A. Cross-Domain Clustering Quality</h3>
<p class="first-p">Table I and Fig. 5 report silhouette, Calinski–Harabasz (size-matched, N = 823), and Davies–Bouldin scores across all six conditions. All silhouette scores fall in the 0.168–0.217 range, which by standard interpretation <a href="#ref-21" class="cite-link">[21]</a> corresponds to weak cluster structure throughout, including the real-only baseline. Every condition, however, clears its respective permutation-null floor by roughly a factor of two (synthetic null 0.087 ± 0.003; real null 0.094 ± 0.008), indicating the discovered structure is not attributable to chance partitioning of noise.</p>

<div class="table-box">
<div class="table-caption">TABLE I<br>CROSS-DOMAIN EXPERIMENTAL RESULTS (5-FOLD USER-LEVEL CV)</div>
<table class="paper-table">
<tr><th>Condition</th><th>Silhouette</th><th>95% CI</th><th>Calinski–Harabasz*</th><th>DBI</th></tr>
<tr><td>Exp. 1: Synth -> Synth</td><td>0.1681</td><td>[0.1654, 0.1698]</td><td>205.4 ± 4.1</td><td>1.848</td></tr>
<tr><td>Exp. 2: Real -> Real</td><td>0.2168</td><td>[0.2051, 0.2242]</td><td>313.7 ± 8.5</td><td>1.537</td></tr>
<tr><td>Exp. 3: Synth -> Real (0-shot)</td><td>0.2117</td><td>[0.1980, 0.2194]</td><td>301.2 ± 9.1</td><td>1.554</td></tr>
<tr><td>Exp. 4a: Ratio 1:1</td><td>0.2127</td><td>[0.2003, 0.2198]</td><td>305.4 ± 7.9</td><td>1.552</td></tr>
<tr><td>Exp. 4b: Ratio 2:1</td><td>0.2122</td><td>[0.1988, 0.2195]</td><td>303.2 ± 8.1</td><td>1.553</td></tr>
<tr><td>Exp. 4c: Ratio 4:1</td><td>0.2118</td><td>[0.1984, 0.2190]</td><td>301.9 ± 8.3</td><td>1.555</td></tr>
</table>
<div style="font-size: 7.5pt; margin-top: 2pt; text-align: left;">*Calinski–Harabasz scores computed using size-matched subsampling (N = 823). DBI = Davies–Bouldin Index.</div>
</div>

<div class="figure-box">
    <img src="{fig5_b64}" alt="Silhouette Results" style="max-height: 1.5in; width: auto; max-width: 100%; object-fit: contain;">
    <div class="figure-caption"><strong>Fig. 5.</strong> Silhouette score (95% CI) across all six experimental conditions, with per-domain permutation-null floors shown in gray. All conditions cluster within a narrow band and exceed their respective noise floor by approximately 2x.</div>
</div>

<h3 class="subsection-title">B. Baseline Comparison: GMM vs. Severity Terciles and K-Means</h3>
<p class="first-p">Table II compares GMM against Severity-Tercile and K-Means baselines. GMM dramatically outperforms Severity Terciles in both Real->Real (silhouette 0.2168 vs. 0.0313, mean diff +0.1855, p_adj &lt; 0.0001, d_z = 34.44) and Zero-Shot Transfer (0.2117 vs. 0.0312, mean diff +0.1805, p_adj &lt; 0.0001, d_z = 24.62). This provides evidence that GMM state discovery captures multivariate structure in the six-dimensional physiological deviation space beyond a univariate severity-quantile partition. Both GMM and K-Means operate on the multivariate deviation representation, and hard-assignment K-Means (K = 3) achieves slightly higher cluster compactness (0.2308 Real->Real; 0.2200 Zero-Shot), consistent with K-Means optimizing spherical sum-of-squares rather than soft Gaussian mixture likelihoods. Crucially, both multivariate clustering approaches substantially outperform the scalar Severity-Tercile baseline.</p>

<div class="table-box">
<div class="table-caption">TABLE II<br>BASELINE COMPARISON ACROSS KEY EVALUATION SETTINGS</div>
<table class="paper-table">
<tr><th>Setting &amp; Model</th><th>Silhouette</th><th>95% CI</th><th>Calinski–Harabasz*</th><th>DBI</th></tr>
<tr><td colspan="5" style="font-weight: bold; background-color: #f1f5f9; text-align: left;">Real -> Real (Exp. 2)</td></tr>
<tr><td>GMM (Primary)</td><td>0.2168</td><td>[0.2051, 0.2242]</td><td>313.7 ± 8.5</td><td>1.537</td></tr>
<tr><td>Severity Tercile Baseline</td><td>0.0313</td><td>[0.0248, 0.0364]</td><td>54.9 ± 3.2</td><td>4.185</td></tr>
<tr><td>K-Means (K = 3) Baseline</td><td>0.2308</td><td>[0.2209, 0.2358]</td><td>326.0 ± 7.4</td><td>1.474</td></tr>
<tr><td colspan="5" style="font-weight: bold; background-color: #f1f5f9; text-align: left;">Synthetic -> Real Zero-Shot (Exp. 3)</td></tr>
<tr><td>GMM (Primary)</td><td>0.2117</td><td>[0.1980, 0.2194]</td><td>301.2 ± 9.1</td><td>1.554</td></tr>
<tr><td>Severity Tercile Baseline</td><td>0.0312</td><td>[0.0271, 0.0346]</td><td>71.7 ± 4.0</td><td>3.680</td></tr>
<tr><td>K-Means (K = 3) Baseline</td><td>0.2200</td><td>[0.2086, 0.2250]</td><td>292.9 ± 8.2</td><td>1.518</td></tr>
</table>
<div style="font-size: 7.5pt; margin-top: 2pt; text-align: left;">*Calinski–Harabasz scores computed using size-matched subsampling (N = 823). DBI = Davies–Bouldin Index.</div>
</div>

<h3 class="subsection-title">C. Zero-Shot Transfer (RQ2)</h3>
<p class="first-p">A GMM trained <em>exclusively</em> on synthetic data, with zero real training rows, achieves silhouette 0.2117 on unseen real test users, against 0.2168 for a model trained directly on real data — a raw difference of -0.0051 (2.37% relative gap; 95% CI [-0.0098, -0.0004]; paired sample effect size d_z = -1.3531). After Nadeau–Bengio variance correction for resampled CV resamples (t = -2.0171, p_adj = 0.1139), the zero-shot transfer difference is not statistically significant, confirming that synthetic-trained models achieve transfer performance on real data comparable to real-trained models. State occupancy distributions remain virtually identical between real-trained (42.2/30.5/27.3%) and zero-shot transfer (42.3/31.0/26.7% for Recovery/Baseline/Strain respectively).</p>

<h3 class="subsection-title">C. Synthetic Augmentation (RQ3)</h3>
<p class="first-p">Table I shows that adding synthetic data to real training data (Exp. 4a–c) does not improve on real-only training (Exp. 2) at any tested ratio; all three ratio conditions show a small negative mean difference relative to Exp. 2, each statistically significant after Holm–Bonferroni correction (p_adj in [0.0021, 0.0038]). Performance is essentially flat across ratios (0.2127 at 1:1 down to 0.2118 at 4:1), proving that synthetic augmentation provides no benefit and produces small performance decreases at all tested ratios when moderate amounts of real-world training data are available (61 eligible users).</p>

<h3 class="subsection-title">D. Domain Shift and a Normalization Caveat</h3>
<p class="first-p">Comparing synthetic and real domains on raw features shows heterogeneous shift: calories_kcal exhibits high shift (Cohen's d = -0.965), avg_hr_day_bpm moderate shift (d = 0.783), and most others low shift. The unnormalized raw deviation sum shows high shift (d = -0.760), while the z-normalized severity score shows a much smaller shift (d = -0.151) (Fig. 6). We flag this explicitly as a normalization artifact: per-user z-scoring rescales feature distributions to zero mean and unit variance, which mechanically compresses between-domain differences regardless of raw physiological match.</p>

<div class="figure-box">
    <img src="{fig6_b64}" alt="Domain Shift Comparison" style="max-height: 1.5in; width: auto; max-width: 100%; object-fit: contain;">
    <div class="figure-caption"><strong>Fig. 6.</strong> Domain-shift comparison across raw features and harmonized severity score (Cohen's d). Z-score normalization acts as a domain harmonization layer.</div>
</div>

<h3 class="subsection-title">E. Temporal Dynamics</h3>
<p class="first-p">The fitted real-domain HMM transition matrix reveals that all three states are self-persistent (diagonal probabilities 0.68–0.76), with Strain showing the highest self-persistence (0.76) and Baseline the lowest (0.68), consistent with the intuition that acute strain episodes tend to persist over consecutive days before resolving.</p>

<h2 class="section-title">VI. EMA External Validation</h2>
<p class="first-p">To evaluate external validity, we tested non-parametric statistical alignment of discovered latent states (<em>Recovery, Baseline, Strain</em>) against 1,751 concurrent daily Ecological Momentary Assessment (EMA) self-report survey entries in the real Fitbit cohort <a href="#ref-15" class="cite-link">[15]</a>. Discovered state occupancies were distributed as <em>Recovery</em> (41.00%), <em>Baseline</em> (39.86%), and <em>Strain</em> (19.14%). Kruskal–Wallis non-parametric ANOVA revealed statistically significant differentiation in self-reported happiness (HAPPY) across latent states (H = 7.0991, p = 0.0287 &lt; 0.05 uncorrected). Days decoded as <em>Recovery</em> exhibited highest self-reported positive affect (0.30 ± 0.46) compared to <em>Baseline</em> (0.24 ± 0.42) and <em>Strain</em> (0.26 ± 0.44). Self-reported tiredness (TIRED) was numerically higher in the <em>Strain</em> state (0.40 ± 0.49) versus <em>Baseline</em> (0.36 ± 0.48), but did not reach statistical significance (H = 1.773, p = 0.4122). All other surveyed EMA dimensions (NEUTRAL, RESTED, TENSE, SAD, ALERT) showed no statistically significant differences (p &gt; 0.10). Crucially, after Holm–Bonferroni correction across the seven EMA outcomes (p_adj = 0.2012), the HAPPY association does not survive multiple-comparison thresholding. We therefore report this as an exploratory uncorrected trend rather than a confirmed external ground-truth correlation.</p>

<div class="table-caption">TABLE III<br>EMA EXTERNAL VALIDATION (N = 1,751 OBSERVATIONS)</div>
<table class="paper-table">
<tr><th>EMA Metric</th><th>Recovery</th><th>Baseline</th><th>Strain</th><th>H (p-value)</th></tr>
<tr><td>HAPPY</td><td><strong>0.30 ± 0.46</strong></td><td>0.24 ± 0.42</td><td>0.26 ± 0.44</td><td><strong>7.099 (p = 0.0287*)</strong></td></tr>
<tr><td>NEUTRAL</td><td>0.29 ± 0.45</td><td>0.32 ± 0.46</td><td>0.25 ± 0.43</td><td>4.450 (p = 0.1081)</td></tr>
<tr><td>TIRED</td><td>0.38 ± 0.49</td><td>0.36 ± 0.48</td><td><strong>0.40 ± 0.49</strong></td><td>1.773 (p = 0.4122)</td></tr>
<tr><td>RESTED</td><td>0.39 ± 0.49</td><td>0.40 ± 0.49</td><td>0.39 ± 0.49</td><td>0.131 (p = 0.9365)</td></tr>
<tr><td>TENSE</td><td>0.22 ± 0.42</td><td>0.23 ± 0.42</td><td>0.21 ± 0.40</td><td>0.593 (p = 0.7434)</td></tr>
<tr><td>SAD</td><td>0.05 ± 0.22</td><td>0.06 ± 0.23</td><td>0.06 ± 0.24</td><td>0.761 (p = 0.6837)</td></tr>
<tr><td>ALERT</td><td>0.13 ± 0.34</td><td>0.15 ± 0.36</td><td>0.15 ± 0.35</td><td>0.613 (p = 0.7361)</td></tr>
</table>

<h2 class="section-title">VII. Discussion and Limitations</h2>
<p class="first-p"><strong>Multivariate structure beyond scalar severity.</strong> Our baseline comparison confirms that GMM state discovery significantly outperforms scalar Severity-Tercile thresholding (silhouette 0.2168 vs. 0.0313, p_adj &lt; 0.0001, d_z = 34.44), providing evidence that personalized state discovery captures multivariate structure in six-dimensional physiological deviation space beyond a univariate severity-quantile partition. Both GMM and K-Means operate on this multivariate representation, and K-Means achieves slightly higher cluster compactness (0.2308 Real->Real; 0.2200 Zero-Shot). This does not invalidate the central result that both multivariate clustering approaches substantially outperform the scalar Severity-Tercile baseline; the contribution of this paper is a methodological and empirical evaluation rather than a claim that GMM is universally superior to K-Means.</p>

<p><strong>Model selection is statistically grounded by BIC/AIC minimization.</strong> BIC and AIC criteria demonstrate a clear minimum at K = 3 among the evaluated candidate components K in {{2, 3, 4, 5}}, matching domain-interpretability requirements for macro-physiological state partitioning.</p>

<p><strong>EMA external validation as an exploratory trend.</strong> Table III summarizes non-parametric alignment against daily EMA self-reports. While the HAPPY metric shows statistically significant differentiation in the uncorrected Kruskal–Wallis analysis (H = 7.0991, p = 0.0287), it does not survive Holm–Bonferroni multiple-comparison correction across the seven EMA outcomes (p_adj = 0.2012). We therefore report this strictly as an exploratory uncorrected trend rather than a confirmed external ground-truth correlation.</p>

<p><strong>Limitations and threats to validity.</strong></p>
<ol style="margin-top: 2pt; margin-bottom: 4pt; padding-left: 1.2em;">
<li><em>Real Cohort Sample Size:</em> The real-domain evaluation relies on 61 HMM-eligible users (4,115 daily timelines) from the LifeSnaps cohort <a href="#ref-15" class="cite-link">[15]</a>, limiting statistical power and generalizability across diverse populations and consumer devices.</li>
<li><em>Exclusion of HRV and SpO2:</em> HRV and SpO2 were excluded from the primary feature vector due to high missingness (&gt;66% missing); this removes potentially informative autonomic stress signals and limits physiological breadth.</li>
<li><em>Weak-Structure Regime:</em> Silhouette values remain in the weak-structure regime (0.16–0.22), reflecting overlapping component boundaries inherent to continuous baseline deviations. Discovered states must not be interpreted as clinically validated physiological categories.</li>
<li><em>Clinical Utility Disclaimer:</em> The Recovery, Baseline, and Strain states are heuristic data-driven proxy labels for exploratory wellness tracking and must not be used as standalone clinical decision variables.</li>
<li><em>Exploratory EMA Validity:</em> The EMA analysis provides exploratory evidence only; self-reported happiness (HAPPY) does not survive Holm–Bonferroni multiple-comparison correction (p_adj = 0.2012).</li>
</ol>

<h2 class="section-title">VIII. Conclusion and Future Work</h2>
<p class="first-p">We presented a personalized, unsupervised, longitudinal wearable health state-discovery framework and evaluated its cross-domain behavior under a leakage-free, statistically rigorous protocol. Latent structure is weak by conventional standards but consistently exceeds permutation-based null baselines across synthetic, real, and transfer conditions; zero-shot synthetic-to-real transfer incurs a small 2.3% performance gap that is not statistically significant after corrected testing (p_adj = 0.1139), and synthetic augmentation does not improve discovery once real training data is moderately sized (61 users). Future work includes extending the harmonized feature space to nocturnal HRV/SpO2, applying factorial HMM extensions <a href="#ref-13" class="cite-link">[13]</a>, and pursuing SHAP-based explainability <a href="#ref-12" class="cite-link">[12]</a>.</p>

<h2 class="section-title">Acknowledgments</h2>
<p class="first-p">The authors thank the Department of Information Science and Engineering at Nitte Meenakshi Institute of Technology for project support. We acknowledge the original LifeSnaps dataset authors (Yfantidou et al., 2022) <a href="#ref-15" class="cite-link">[15]</a> for curating and releasing the multi-modal longitudinal dataset.</p>

<div style="break-before: page; page-break-before: always;"></div>
<h2 class="section-title">References</h2>
<div class="references">
    <div id="ref-1" class="ref-item">[1] N. Rashid, T. Mortlock, and M. A. Al Faruque, "Stress detection using context-aware sensor fusion from wearable devices," <em>IEEE Internet of Things Journal</em>, vol. 10, no. 18, pp. 16120–16132, 2023.</div>
    <div id="ref-2" class="ref-item">[2] Y. Chen, J. Wang, C. Yu, W. Gao, and X. Qin, "FedHealth: A federated transfer learning framework for wearable healthcare," <em>IEEE Intelligent Systems</em>, vol. 35, no. 4, pp. 83–93, 2020.</div>
    <div id="ref-3" class="ref-item">[3] H. S. Saad, J. F. W. Zaki, and M. M. Abdelsalam, "Employing of machine learning and wearable devices in healthcare system: tasks and challenges," <em>Neural Computing and Applications</em>, vol. 36, pp. 17829–17849, 2024.</div>
    <div id="ref-4" class="ref-item">[4] F. Sabry, T. Eltaras, W. Labda, K. Alzoubi, and Q. Malluhi, "Machine learning for healthcare wearable devices: The big picture," <em>Journal of Healthcare Engineering</em>, vol. 2022, Art. ID 4653923, 2022.</div>
    <div id="ref-5" class="ref-item">[5] M. Dunn, R. Runge, and M. Snyder, "Wearables and the medical cloud: Real-time physiological monitoring and anomaly detection," <em>Cell Reports Medicine</em>, vol. 2, no. 1, p. 100178, 2021.</div>
    <div id="ref-6" class="ref-item">[6] A. Seshadri, B. R. Kociuga, and J. A. Rogers, "Wearable sensors for global health monitoring: A review," <em>IEEE Trans. Biomed. Eng.</em>, vol. 66, no. 5, pp. 1240–1253, 2019.</div>
    <div id="ref-7" class="ref-item">[7] E. J. Topol, "High-performance medicine: the convergence of human and artificial intelligence," <em>Nature Medicine</em>, vol. 25, no. 1, pp. 44–56, 2019.</div>
    <div id="ref-8" class="ref-item">[8] X. Li, J. Dunn, and M. Snyder, "Digital health: tracking physiomes and activity using wearable sensors," <em>PLOS Biology</em>, vol. 15, no. 1, p. e2001402, 2017.</div>
    <div id="ref-9" class="ref-item">[9] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," <em>Patterns</em>, vol. 4, no. 9, p. 100804, 2023.</div>
    <div id="ref-10" class="ref-item">[10] J. A. Rogers, T. R. Ray, and M. Choi, "Continuous wireless monitoring with soft bioelectronic systems," <em>Nature Medicine</em>, vol. 27, no. 4, pp. 595–608, 2021.</div>
    <div id="ref-11" class="ref-item">[11] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," <em>JMLR</em>, vol. 12, pp. 2825–2830, 2011.</div>
    <div id="ref-12" class="ref-item">[12] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in <em>Proc. NeurIPS</em>, vol. 30, pp. 4765–4774, 2017.</div>
    <div id="ref-13" class="ref-item">[13] Z. Ghahramani and M. I. Jordan, "Factorial Hidden Markov Models," <em>Machine Learning</em>, vol. 29, no. 2, pp. 245–273, 1997.</div>
    <div id="ref-14" class="ref-item">[14] L. R. Rabiner, "A tutorial on Hidden Markov Models and selected applications in speech recognition," <em>Proc. IEEE</em>, vol. 77, no. 2, pp. 257–286, 1989.</div>
    <div id="ref-15" class="ref-item">[15] S. Yfantidou et al., "LifeSnaps, a 4-month multi-modal dataset capturing unobtrusive snapshots of our lives in the wild," <em>Scientific Data</em>, vol. 9, no. 663, 2022.</div>
    <div id="ref-16" class="ref-item">[16] D. A. Reynolds, "Gaussian Mixture Models," in <em>Encyclopedia of Biometrics</em>, Springer, pp. 659–663, 2009.</div>
    <div id="ref-17" class="ref-item">[17] M. H. DeGroot and M. J. Schervish, <em>Probability and Statistics</em>, 4th ed., Pearson, 2012.</div>
    <div id="ref-18" class="ref-item">[18] G. Schwarz, "Estimating the dimension of a model," <em>Annals of Statistics</em>, vol. 6, no. 2, pp. 461–464, 1978.</div>
    <div id="ref-19" class="ref-item">[19] C. Nadeau and Y. Bengio, "Inference for the generalization error," <em>Machine Learning</em>, vol. 52, no. 3, pp. 239–281, 2003.</div>
    <div id="ref-20" class="ref-item">[20] S. Holm, "A simple sequentially rejective multiple test procedure," <em>Scand. J. Stat.</em>, vol. 6, no. 2, pp. 65–70, 1979.</div>
    <div id="ref-21" class="ref-item">[21] L. Kaufman and P. J. Rousseeuw, <em>Finding Groups in Data: An Introduction to Cluster Analysis</em>, Wiley, 1990.</div>
</div>

</div>

</body>
</html>
"""

    print(f"Writing Base64-embedded HTML to {HTML_PATH}...")
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("Compiling PDFs via Edge Headless...")
    msedge_cmd = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    
    cmd_out = [
        msedge_cmd, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}", f"file:///{HTML_PATH.as_posix()}"
    ]
    subprocess.run(cmd_out, check=True)
    
    cmd_root = [
        msedge_cmd, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={ROOT_PDF}", f"file:///{HTML_PATH.as_posix()}"
    ]
    subprocess.run(cmd_root, check=True)
    
    print("Successfully compiled PDFs with Base64 embedded figures!")

if __name__ == "__main__":
    main()
