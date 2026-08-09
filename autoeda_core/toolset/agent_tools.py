
import seaborn as sns

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from ..profiler import is_non_distributional_column
from ..llm_config import get_api_key, get_model, get_base_url

sns.set_theme(style="whitegrid")


from .utils import _sanitize_col_name, _safe_float, _is_numeric_col


def ask_clarifying_question(df: pd.DataFrame, question: str, **kwargs) -> Dict[str, Any]:
    return {"question": question, "status": "paused_for_user_input"}


class AskClarifyingQuestionArgs(BaseModel):
    question: str = Field(description="The question to ask the user to clarify ambiguity.")

