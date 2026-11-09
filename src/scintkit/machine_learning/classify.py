"""Label gated satellite-minutes as Multipath, RFI or Scintillation with the bundled model.

Needs the ``ml`` extra (scikit-learn, joblib); without it every minute is "Unclassified".
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
UNCLASSIFIED = "Unclassified"  # ml_class of every minute the model did not label
# The model labels a featured minute only above this channel-1 S4 (the trained feature gate is 0.15).
ML_CLASSIFY_MIN_S4 = 0.2
TEC_POWER = "tec12_power_db_1_10_hz"  # NaN when the minute has no usable dual-frequency TEC


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
    """Model ``ml_class`` and class probabilities for every row of a :func:`compute_features` table."""
    model = load_model()
    out = features[["prn", "datetime"]].copy()
    if features.empty:
        return unclassified(out)
    probabilities = model.predict_proba(features[FEATURES])
    out["ml_class"] = pd.array(model.classes_[probabilities.argmax(axis=1)], dtype="string")
    for k, name in enumerate(model.classes_):
        out[PROBABILITY_COLUMNS[name]] = probabilities[:, k].astype(np.float32)
    return out


def classifiable(features):
    """Rows the model labels: S4 >= ML_CLASSIFY_MIN_S4 and both frequencies (S4 and TEC of channel 2) present."""
    dual = features["s4_2"].notna() & features[TEC_POWER].notna()
    return features[features["s4_1"].ge(ML_CLASSIFY_MIN_S4) & dual]


def unclassified(minute):
    """``minute`` with ``ml_class`` = UNCLASSIFIED and NaN probabilities."""
    minute = minute.copy()
    minute["ml_class"] = pd.array([UNCLASSIFIED] * len(minute), dtype="string")
    for column in PROBABILITY_COLUMNS.values():
        minute[column] = np.float32(np.nan)
    return minute


def classify_minutes(frame, minute, fs):
    """``(minute with ML_COLUMNS, lvl3 metadata)``; minutes not sent to the model are UNCLASSIFIED.

    ``frame`` is the full-rate frame and ``fs`` its rate (Hz). Never raises: when
    the data are not 20 Hz, scikit-learn is missing or anything fails, one message
    is logged and every minute is UNCLASSIFIED.
    """
    if fs != FS_HZ:
        reason = f"skipped: the model needs {FS_HZ} Hz data, got {fs:g} Hz"
    else:
        try:
            labels = classify(classifiable(compute_features(frame)))
            out = unclassified(minute).set_index(["prn", "datetime"])
            out.update(labels.set_index(["prn", "datetime"]))
            out = out.reset_index()[list(minute.columns) + ML_COLUMNS]
            logger.info("classified %d satellite-minutes: %s", len(labels),
                        labels["ml_class"].value_counts().to_dict())
            return out, {"ml_classification": "on", "ml_classifier_ran": "true", **model_metadata()}
        except ImportError as error:
            reason = f"skipped: scikit-learn is not installed or not importable (pip install 'scintkit[ml]'): {error}"
        except Exception as error:  # the classifier must never fail the file
            logger.exception("classification failed")
            reason = f"failed: {type(error).__name__}: {error}"
    logger.warning("signal classification %s", reason)
    return unclassified(minute), {"ml_classification": reason, "ml_classifier_ran": "false"}
