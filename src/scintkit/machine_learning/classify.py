"""Label gated satellite-minutes as Multipath, RFI or Scintillation with the bundled model.

Needs the ``ml`` extra (scikit-learn, joblib); without it the lvl3 columns stay null.
"""

import functools
import hashlib
import logging
from importlib.resources import as_file, files

import numpy as np
import pandas as pd

from scintkit.machine_learning.features import FEATURES, FS_HZ, compute_features

logger = logging.getLogger(__name__)

MODEL_FILE = "histgb_full_25features.joblib"
# lvl3 column holding the probability of each model class.
PROBABILITY_COLUMNS = {
    "Multipath": "ml_prob_multipath",
    "RFI": "ml_prob_rfi",
    "Scintillation": "ml_prob_scintillation",
}
ML_COLUMNS = ["ml_class", *PROBABILITY_COLUMNS.values()]


@functools.cache
def load_model():
    """The bundled HistGradientBoostingClassifier, loaded once per process; ImportError without scikit-learn."""
    import joblib
    import sklearn  # noqa: F401  (joblib alone cannot unpickle the model)

    with as_file(files("scintkit") / "data" / MODEL_FILE) as path:
        bundle = joblib.load(path)
    if list(bundle["features"]) != FEATURES or list(bundle["model"].classes_) != list(PROBABILITY_COLUMNS):
        raise ValueError(f"{MODEL_FILE} does not match the feature or class list of this scintkit")
    return bundle["model"]


@functools.cache
def model_metadata():
    """lvl3 file metadata naming the model: file name and the first 12 hex digits of its SHA-256."""
    digest = hashlib.sha256((files("scintkit") / "data" / MODEL_FILE).read_bytes()).hexdigest()
    return {"ml_model": MODEL_FILE, "ml_model_sha256": digest[:12]}


def classify(features):
    """``ml_class`` and the class probabilities for each row of a :func:`compute_features` table."""
    model = load_model()
    out = features[["prn", "datetime"]].copy()
    if features.empty:
        return null_classes(out)
    probabilities = model.predict_proba(features[FEATURES])
    out["ml_class"] = pd.array(model.classes_[probabilities.argmax(axis=1)], dtype="string")
    for k, name in enumerate(model.classes_):
        out[PROBABILITY_COLUMNS[name]] = probabilities[:, k].astype(np.float32)
    return out


def null_classes(minute):
    """``minute`` with the ML columns present and null."""
    minute = minute.copy()
    minute["ml_class"] = pd.array([pd.NA] * len(minute), dtype="string")
    for column in PROBABILITY_COLUMNS.values():
        minute[column] = np.float32(np.nan)
    return minute


def classify_minutes(frame, minute, fs):
    """``(minute with ML_COLUMNS, lvl3 metadata)``; columns stay null where a minute is not gated.

    ``frame`` is the full-rate frame and ``fs`` its rate (Hz). Never raises: when
    the data are not 20 Hz, scikit-learn is missing or anything fails, one message
    is logged and all ML columns are null.
    """
    if fs != FS_HZ:
        reason = f"skipped: the model needs {FS_HZ} Hz data, got {fs:g} Hz"
    else:
        try:
            labels = classify(compute_features(frame))
            out = minute.merge(labels, on=["prn", "datetime"], how="left")
            logger.info("classified %d satellite-minutes: %s", len(labels),
                        labels["ml_class"].value_counts().to_dict())
            return out, {"ml_classification": "on", **model_metadata()}
        except ImportError as error:
            reason = f"skipped: scikit-learn is not installed (pip install 'scintkit[ml]'): {error}"
        except Exception as error:  # the classifier must never fail the file
            logger.exception("classification failed")
            reason = f"failed: {type(error).__name__}: {error}"
    logger.warning("signal classification %s", reason)
    return null_classes(minute), {"ml_classification": reason}
