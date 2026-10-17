# Model Card: Breast Cancer Classifier

## Intended use
Educational demonstration of an MLOps pipeline. **Not a medical device and not for diagnosis, screening, or
any clinical decision.** The model has not been validated on patients or reviewed by clinicians.

## Model
Logistic regression (C=1.0, max 1000 iterations) after standard scaling, in a scikit-learn pipeline.
Output: probability that a tumour is malignant, converted to a label with a tuned threshold.

## Data
Wisconsin Diagnostic Breast Cancer dataset, loaded from `sklearn.datasets.load_breast_cancer`.
569 samples, 30 numeric features computed from digitised images of fine-needle aspirates
(mean, standard error and worst value of 10 cell-nucleus measurements). About 37% malignant.
Split 80/20, stratified by label (455 train, 114 test, seed 42).

## Training and threshold
The decision threshold is chosen on out-of-fold predictions from 5-fold stratified cross-validation on the
training set: the highest threshold that still achieves recall of at least 0.97 on malignant cases.
Recall was prioritised because a missed malignancy is the costlier mistake; the price is more false positives.
The current threshold is about 0.357. The test set was not used for model or threshold selection.

## Metrics (held-out test set, n=114, threshold applied)

| Metric | Value |
|---|---|
| Recall (malignant) | 0.976 |
| Precision (malignant) | 0.976 |
| ROC AUC | 0.997 |
| Brier score | 0.019 |
| False negatives | 1 |
| False positives | 1 |

With only 114 test rows, each error moves recall and precision by roughly 2 to 3 percentage points,
so these numbers carry wide uncertainty. Cross-validation recall (0.971) and test recall (0.976) agree.

## Quality gate
The pipeline fails if test recall < 0.95, precision < 0.90, or ROC AUC < 0.98 (thresholds in `params.yaml`).

## Limitations
- Small dataset (569 samples) from a single, well-known benchmark; no external or prospective validation.
- Features come from one imaging and measurement protocol. Different equipment or populations may shift the
  feature distributions and degrade performance; drift monitoring exists as a script but is not automated.
- The model needs all 30 features and does not check whether inputs are physically plausible.
- The demo API logs requests only to temporary storage on Lambda, so there is no durable prediction history.
- No fairness or subgroup analysis was done; the dataset has no demographic information.

## Ethical considerations
Presenting a probability for a cancer diagnosis can cause harm if misread. The web UI and API responses
carry an "educational demo, not for medical use" notice, and the model should not be used on real patient data.
