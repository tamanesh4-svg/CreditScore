"""
Workaround loader for credit_risk_model_final.pkl.

Normal unpickling calls XGBoosterUnserializeFromBuffer, which is what raises
"input stream corrupted". Each pickled booster is a UBJSON buffer holding a
"Config" section and a "Model" section. This loader keeps only the "Model"
section and loads it with Booster.load_model(), the normal, version-stable path.

Use:
    from safe_load import load_model
    model = load_model("credit_risk_model_final.pkl")
"""
from pathlib import Path

from joblib.numpy_pickle import NumpyUnpickler
from xgboost import Booster

# UBJSON object key as XGBoost writes it: 'L' + 8-byte big-endian length + name
_MODEL_KEY = b"L" + (5).to_bytes(8, "big") + b"Model"


class SafeBooster(Booster):
    def __setstate__(self, state):
        raw = bytes(state["handle"])
        pos = raw.find(_MODEL_KEY)
        if pos < 0:
            raise ValueError("No 'Model' section found in pickled booster")
        start = pos + len(_MODEL_KEY)
        if raw[start:start + 1] != b"{":
            raise ValueError("Unexpected layout of pickled booster")
        model_bytes = raw[start:-1]  # drop the closing brace of the root object
        Booster.__init__(self)
        self.load_model(bytearray(model_bytes))


class _SafeUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if module == "xgboost.core" and name == "Booster":
            return SafeBooster
        return super().find_class(module, name)


def load_model(path):
    path = Path(path)
    with open(path, "rb") as f:
        return _SafeUnpickler(str(path), f, ensure_native_byte_order=True).load()


if __name__ == "__main__":
    import sys

    m = load_model(sys.argv[1] if len(sys.argv) > 1 else "credit_risk_model_final.pkl")
    print("Loaded:", type(m))
