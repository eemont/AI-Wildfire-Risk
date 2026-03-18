# AI Module – Wildfire Project

## Overview

This folder contains the machine learning pipeline for processing wildfire detection data and generating predictions. The current implementation focuses on building a baseline model for classifying high-confidence fire detections using NASA FIRMS and NOAA data.

## Key Improvements

### 1. Added Pytest Tests

* **Feature Engineering Tests**

  * Validate `confidence_to_bin()` mapping for FIRMS (`n/h`) and NOAA (`medium`, etc.)
  * Ensure feature matrix contains expected columns

* **Model Persistence Tests**

  * Verify model save/load functionality via `model_store.py`

These tests can be run with:

```bash
cd ai/src
python -m pytest ai_wildfire/tests -q
```

---

### 2. Artifact Management (metrics + models)

Generated artifacts are stored in:

```
ai/artifacts/
```

This includes:

* `baseline_model.joblib`
* `metrics.json`
* model metadata

**Important:**

* These files are **not committed to git**
* Added to `.gitignore` to avoid versioning generated outputs


### 3. Clear Label Definition

The binary label is defined in `features.py`:

1 → high-confidence fire detection  
0 → non-high-confidence detection

Supported mappings:

* FIRMS: `h → 1`, `n → 0`
* NOAA: `high → 1`, `medium/low/nominal → 0`

This is documented directly in the code with a detailed docstring.

**Note:**
This is a baseline classification task, not a true wildfire risk prediction model.


### 4. Unified Feature Pipeline

Previously, feature logic was duplicated between training and prediction.

Now:

* `build_feature_matrix()` is used in both:

  * `train.py`
  * `predict.py`



### 5. Centralized Configuration (No Hardcoded Paths)

All paths are now defined in `configs.py`:

* `DB_PATH`
* `ARTIFACT_DIR`
* `MODEL_FILENAME`
* `METRICS_FILENAME`

Both training and prediction use these shared values.

Benefits:

* No duplicated path logic
* Works across environments (local, Docker, CI)
* Prevents incorrect DB loading

---

## How to Run

### Train Model

```bash
cd ai/src
python -m ai_wildfire.train --limit 500
```

### Run Predictions

```bash
python -c "from ai_wildfire.predict import predict_from_db; predict_from_db(limit=10)"
```

prediction score is the model’s estimated probability that a detection is high-confidence (label = 1)
---

## Current Limitations

* Highly imbalanced dataset (few `high` confidence samples)
* Model currently predicts detection confidence, not ignition risk
* Limited feature set (no weather or environmental data yet)

---

## Next Steps

* Add weather and environmental data sources
* Build grid-based wildfire risk prediction model
* Improve class balancing and evaluation metrics
* Add CI pipeline for automated testing

---

## Summary

This update addresses all requested feedback:

✔ Added meaningful pytest coverage
✔ Defined and documented label behavior
✔ Cleaned up duplicated logic
✔ Centralized configuration
✔ Clarified artifact handling

The AI module is now more maintainable, testable, and ready for extension.
