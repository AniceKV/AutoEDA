import json
import unittest

import numpy as np
import pandas as pd

from autoeda_core.toolset.utils import _json_safe


class JsonSerializationTests(unittest.TestCase):
    def test_json_safe_converts_numpy_and_pandas_scalars(self):
        payload = {
            "int64": np.int64(3),
            "float64": np.float64(1.25),
            "bool_": np.bool_(True),
            "array": np.array([1, np.int64(2)]),
            "timestamp": pd.Timestamp("2024-01-02 03:04:05"),
            "nested": [{"value": np.int64(4)}],
        }

        safe_payload = _json_safe(payload)

        self.assertEqual(safe_payload["int64"], 3)
        self.assertEqual(safe_payload["float64"], 1.25)
        self.assertTrue(safe_payload["bool_"])
        self.assertEqual(safe_payload["array"], [1, 2])
        self.assertEqual(safe_payload["timestamp"], "2024-01-02T03:04:05")
        self.assertEqual(safe_payload["nested"][0]["value"], 4)

        json.dumps(safe_payload)


if __name__ == "__main__":
    unittest.main()
