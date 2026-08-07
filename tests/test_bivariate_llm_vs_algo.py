import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autoeda_core import tools

def test_algorithmic_fallback_when_no_pairs_provided():
    # Create a simple DataFrame
    df = pd.DataFrame({
        "num_col1": [1.5, 2.3, 3.1, 4.8, 5.2, 6.9, 7.1, 8.4, 9.6, 10.2, 11.7, 12.0],
        "num_col2": [10, 25, 30, 45, 50, 65, 70, 85, 90, 105, 110, 125],
        "cat_col1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
    })
    
    # 1. Test plot_semantic_bivariate_relationships with no pairs (should use algorithmic fallback)
    result = tools.plot_semantic_bivariate_relationships(df, bivariate_pairs=None)
    assert result["count"] > 0
    # The fallback should select pairs from numeric/discrete columns
    pairs = result["bivariate_data"]
    assert any(p["rationale"].startswith("Categorical breakdown") or 
               p["rationale"].startswith("Segmented distribution") or
               p["rationale"].startswith("Bivariate scatter comparison") for p in pairs)

def test_llm_driven_custom_pairs():
    df = pd.DataFrame({
        "num_col1": [1.5, 2.3, 3.1, 4.8, 5.2, 6.9, 7.1, 8.4, 9.6, 10.2, 11.7, 12.0],
        "num_col2": [10, 25, 30, 45, 50, 65, 70, 85, 90, 105, 110, 125],
        "cat_col1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
    })
    
    # 2. Test plot_semantic_bivariate_relationships with custom (simulated LLM-selected) pairs
    custom_pairs = [
        {"x": "num_col1", "y": "num_col2", "rationale": "Verify linear scale comparison"}
    ]
    result = tools.plot_semantic_bivariate_relationships(df, bivariate_pairs=custom_pairs)
    assert result["count"] == 1
    assert result["bivariate_data"][0]["x"] == "num_col1"
    assert result["bivariate_data"][0]["y"] == "num_col2"
    assert result["bivariate_data"][0]["rationale"] == "Verify linear scale comparison"

@patch("openai.OpenAI")
def test_infer_llm_bivariate_pairs_api_flow(mock_openai_class):
    # Setup mock response for OpenRouter API
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_choice = MagicMock()
    mock_choice.message.content = """
    [
        {"feature_1": "age", "feature_2": "income", "rationale": "Check life-cycle income hypothesis"}
    ]
    """
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 70],
        "income": [1000, 2000, 3000, 4000, 5000, 6000]
    })
    
    # Temporarily set API key to force LLM path
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"}):
        pairs = tools.infer_llm_bivariate_pairs(df, dataset_name="test_dataset")
        
        assert len(pairs) == 1
        assert pairs[0]["feature_1"] == "age"
        assert pairs[0]["feature_2"] == "income"
        assert pairs[0]["rationale"] == "Check life-cycle income hypothesis"
        assert pairs[0]["source"] == "llm_inferred"
