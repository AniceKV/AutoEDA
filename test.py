from autoeda_core import AutoEDAEngine
from dotenv import load_dotenv

load_dotenv()

engine = AutoEDAEngine()


results = engine.analyze(
    data_path="test_data/StudentsPerformance.csv",
    user_request="Perform full exploratory data analysis",
    workspace_dir="./my_analysis_output",
    answer_fn=lambda question: "infer it yourself",
)