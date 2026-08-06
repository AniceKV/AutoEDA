import pytest
import pandas as pd
import copy
from tools import StatefulDataStore  # Test your actual tools implementation!

def test_initial_state_setup():
    store = StatefulDataStore()
    df = pd.DataFrame({"A": [1, 2, 3]})
    state = {"imputation_res": None, "engineered_res": []}
    
    result_str = store.set_initial_state(df, state)
    
    assert result_str == "memory:v0"
    assert store.version == 0
    assert len(store.history) == 1
    assert store.history[0]["rows"] == 3
    assert store.history[0]["cols"] == 1
    assert store.history[0]["agent_state"] == state
    assert store.history[0]["action"] == "initial_load"

def test_save_checkpoint():
    store = StatefulDataStore()
    df = pd.DataFrame({"A": [1, 2, 3]})
    state = {"imputation_res": None}
    
    store.set_initial_state(df, state)
    
    # Modify data and state for next checkpoint
    df_new = df.copy()
    df_new["B"] = [4, 5, 6]
    state_new = {"imputation_res": "completed"}
    
    checkpoint_str = store.save_checkpoint(df_new, state_new, "impute_missing_data")
    
    assert checkpoint_str == "memory:v1"
    assert store.version == 1
    assert len(store.history) == 2
    assert store.history[1]["rows"] == 3
    assert store.history[1]["cols"] == 2
    assert store.history[1]["agent_state"] == state_new
    assert store.history[1]["action"] == "impute_missing_data"

def test_invalid_checkpoint_raises_error():
    store = StatefulDataStore()
    df = pd.DataFrame({"A": [1, 2, 3]})
    store.set_initial_state(df, {})
    
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Cannot checkpoint invalid or empty DataFrame"):
        store.save_checkpoint(empty_df, {}, "empty_step")

def test_single_step_rollback():
    store = StatefulDataStore()
    df_v0 = pd.DataFrame({"A": [1, 2, 3]})
    state_v0 = {"engineered_res": []}
    store.set_initial_state(df_v0, state_v0)
    
    df_v1 = df_v0.copy()
    df_v1["B"] = [4, 5, 6]
    state_v1 = {"engineered_res": ["feature_B"]}
    store.save_checkpoint(df_v1, state_v1, "engineer_features")
    
    # Act: rollback from v1 to v0
    restored_df, restored_state, version = store.rollback()
    
    assert version == 0
    assert store.version == 0
    assert len(store.history) == 1
    assert "B" not in restored_df.columns
    assert restored_state["engineered_res"] == []

def test_rollback_boundary():
    store = StatefulDataStore()
    df = pd.DataFrame({"A": [1, 2, 3]})
    state = {"status": "init"}
    store.set_initial_state(df, state)
    
    # Try rolling back when only v0 exists
    restored_df, restored_state, version = store.rollback()
    
    assert version == 0
    assert store.version == 0
    assert len(store.history) == 1
    assert restored_state["status"] == "init"

def test_metadata_deep_copy_integrity():
    store = StatefulDataStore()
    df = pd.DataFrame({"A": [1, 2, 3]})
    state_v0 = {"engineered_res": []}
    store.set_initial_state(df, state_v0)
    
    # Modify the state dictionary in place during step 1 execution
    state_v1 = {"engineered_res": ["feature_B"]}
    store.save_checkpoint(df, state_v1, "step_1")
    
    # Modify state_v1's nested structures *after* checkpointing
    # This simulates agent loop mutating dictionary fields on a step that eventually crashes
    state_v1["engineered_res"].append("corrupted_feature_C")
    
    # Rollback to v0
    _, restored_state, _ = store.rollback()
    
    # Ensure nested list in v0 is completely untainted
    assert restored_state["engineered_res"] == []

def test_purge_intermediate_states():
    store = StatefulDataStore()
    df = pd.DataFrame({"A": [1, 2, 3]})
    store.set_initial_state(df, {"v": 0})
    
    store.save_checkpoint(df, {"v": 1}, "step_1")
    store.save_checkpoint(df, {"v": 2}, "step_2")
    store.save_checkpoint(df, {"v": 3}, "step_3")
    
    assert len(store.history) == 4
    
    store.purge_intermediate_states()
    
    # Should only retain v0 (initial) and v3 (final)
    assert len(store.history) == 2
    assert store.history[0]["agent_state"] == {"v": 0}
    assert store.history[1]["agent_state"] == {"v": 3}
