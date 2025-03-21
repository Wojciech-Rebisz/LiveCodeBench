import pathlib

from lcb_runner.lm_styles import LanguageModel, LMStyle
from lcb_runner.utils.scenarios import Scenario


def ensure_dir(path: str, is_file=True):
    if is_file:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def get_cache_path(model_repr: str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    path = f"cache/{model_repr}/{scenario}_{n}_{temperature}.json"
    ensure_dir(path)
    return path


def get_output_path(model_repr: str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    path = f"output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}.json"
    ensure_dir(path)
    return path


def get_eval_all_output_path(model_repr: str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    path = f"output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}_eval_all.json"
    return path


def save_summary_csv(output_path, runner, args, metrics=None):
    import pandas as pd

    pass_val = metrics[0].get("pass@1") if metrics else None
    params = runner.generate_params
    params["n_samples"] = args.n
    if args.cot_code_execution:
        params["cot_code_execution"] = True

    if args.scenario == Scenario.selfrepair:
        task = "code-fixing"
    else:
        task = "code-generation"

    data = {
        "dataset": [args.scenario.value],
        "language": ["python"],
        "model": [args.model],
        "framework": ["LiveCodeBench"],
        "score_name": ["pass@1"],
        "score": [pass_val],
        "task": [task],
        "generation_parameters": [params],
        "all_scores": [{"pass@1": pass_val}],
        "inference_env": ["wml"],
        "code_execution_env": ["wxai-code-exec"],
        "is_from_academic_paper": [False],
    }

    df = pd.DataFrame(data)

    df.to_csv(output_path, index=False)
