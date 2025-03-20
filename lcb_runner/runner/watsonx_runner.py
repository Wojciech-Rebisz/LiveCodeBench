import os
from time import sleep
import logging

try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner
from lcb_runner.lm_styles import LanguageModel


class WatsonxRunner(BaseRunner):

    def __init__(self, args, model: LanguageModel):
        super().__init__(args, model)
        self.generate_params: dict[str | str] = {
            "temperature": args.temperature,
            "max_new_tokens": args.max_tokens,
            "top_p": args.top_p,
            "random_seed": args.seed,
        }
        creds = self._read_wx_credentials()

        self.client = APIClient(
            credentials=creds,
            space_id=creds.get("space_id"),
            project_id=creds.get("project_id"),
        )

        self.model_inference = ModelInference(
            model_id=args.model,
            api_client=self.client,
            params=self.generate_params,
        )

    @staticmethod
    def _read_wx_credentials() -> dict[str, str]:
        credentials = {}

        url = os.environ.get("WX_URL")
        if not url:
            raise EnvironmentError(
                "You need to specify the URL address by setting the env "
                "variable 'WX_URL', if you want to run watsonx.ai inference."
            )
        credentials["url"] = url

        project_id = os.environ.get("WX_PROJECT_ID")
        space_id = os.environ.get("WX_SPACE_ID")
        if project_id and space_id:
            logging.warning(
                "Both the project ID and the space ID were specified. "
                "The class 'WxInference' will access the project by default."
            )
            credentials["project_id"] = project_id
        elif project_id:
            credentials["project_id"] = project_id
        elif space_id:
            credentials["space_id"] = space_id
        else:
            raise EnvironmentError(
                "You need to specify the project ID or the space id by setting the "
                "appropriate env variable (either 'WX_PROJECT_ID' or 'WX_SPACE_ID'), "
                "if you want to run watsonx.ai inference."
            )

        apikey = os.environ.get("WX_APIKEY")
        username = os.environ.get("WX_USERNAME")
        password = os.environ.get("WX_PASSWORD")
        if apikey and username and password:
            logging.warning(
                "All of API key, username and password were specified. "
                "The class 'WxInference' will use the API key for authorization "
                "by default."
            )
            credentials["apikey"] = apikey
        elif apikey:
            credentials["apikey"] = apikey
        elif username and password:
            credentials["username"] = username
            credentials["password"] = password
        else:
            raise EnvironmentError(
                "You need to specify either the API key, or both the username and "
                "password by setting appropriate env variable ('WX_APIKEY', 'WX_USERNAME', "
                "'WX_PASSWORD'), if you want to run watsonx.ai inference."
            )

        return credentials

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
