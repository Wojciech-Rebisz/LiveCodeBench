import os
from time import sleep

try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner
from lcb_runner.lm_styles import LanguageModel


class WatsonxRunner(BaseRunner):

    client = APIClient(
        credentials=Credentials(
            url=os.getenv("WML_URL"), api_key=os.getenv("WML_APIKEY")
        ),
        space_id=os.getenv("WML_SPACE_ID"),
        project_id=os.getenv("WML_PROJECT_ID"),
    )

    def __init__(self, args, model: LanguageModel):
        super().__init__(args, model)
        self.generate_params: dict[str | str] = {
            "temperature": args.temperature,
            "max_new_tokens": args.max_tokens,
            "top_p": args.top_p,
            "random_seed": 10,
        }

        self.model_inference = ModelInference(
            model_id=args.model,
            api_client=WatsonxRunner.client,
            params=self.generate_params,
        )

    def _run_single(self, prompt: str | list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, str) or isinstance(prompt, list)

        def __run_single(counter) -> list[str]:
            try:
                prompts = [prompt] * self.args.n
                outputs = self.model_inference.generate(prompt=prompts)
                if isinstance(outputs, list):
                    return [val["results"][0]["generated_text"] for val in outputs]

                return [outputs["results"][0]["generated_text"]]
            except Exception as e:
                sleep_time = 20 * (11 - counter)
                print("Exception: ", repr(e), f"Sleeping for {sleep_time} seconds...")
                sleep(sleep_time)
                counter = counter - 1
                if counter == 0:
                    print(f"Failed to run model for {prompt}!")
                    print("Exception: ", repr(e))
                    raise e
                return __run_single(counter)

        try:
            return __run_single(10)
        except Exception as e:
            raise e
