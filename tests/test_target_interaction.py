import pandas as pd
import numpy as np
from autoeda_core.toolset.data_visualizer import default_visualizer, plot_target_interaction


def test_multi_feature_target_interaction():
    np.random.seed(42)
    df = pd.DataFrame({
        "target": np.random.choice(["A", "B"], size=100),
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "feat3": np.random.choice(["X", "Y", "Z"], size=100),
        "feat4": np.random.randint(1, 100, size=100),
        "feat5": np.random.randn(100),
        "feat6": np.random.randn(100),
        "feat7": np.random.randn(100),
        "feat8": np.random.randn(100),
        "feat9": np.random.randn(100),
        "feat10": np.random.randn(100),
        "feat11": np.random.randn(100),
    })

    res = default_visualizer.plot_target_interaction(df, target_col="target", top_n=10)
    assert res["target_col"] == "target"
    assert "target_interactions" in res
    assert len(res["target_interactions"]) == 10
    assert "interaction_data" in res
    assert res["interaction_data"]["target_col"] == "target"
    assert len(res["top_features"]) == 10
