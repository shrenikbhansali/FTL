import os

from federatedscope.core.data.utils import download_url

IFEVAL_URL = (
    "https://huggingface.co/datasets/google/IFEval/"
    "resolve/main/ifeval_input_data.jsonl"
)
IFEVAL_FILE = "ifeval_input_data.jsonl"


def download_ifeval(destination_dir="data"):
    os.makedirs(destination_dir, exist_ok=True)
    download_url(IFEVAL_URL, destination_dir)


if __name__ == "__main__":
    download_ifeval("data")
