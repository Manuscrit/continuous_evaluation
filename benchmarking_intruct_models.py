import logging

from benchmarking_pythia_models import benchmark, benchmark_pythia_models
from models import INSTRUCT_MODELS
from shared import DEBUG


def benchmark_instruct_models():
    checkpoints_selected = [None]
    logging.info(f"checkpoints_selected: {checkpoints_selected}")

    # models_selected = INSTRUCT_MODELS
    models_selected = [
        m for m in INSTRUCT_MODELS if "orca_mini" in m or "vicuna" in m
    ][0:]
    # models_selected = ["ichitaka/falcon-40b-instruct-8bit"]
    logging.info(f"models_selected: {models_selected}")

    benchmark(
        benchmark_kwargs={"n_statements": 2 if DEBUG else 500},
        gene_kwargs={
            "extra_tokens": 1,
            "temperature": 1.0,
            "do_sample": True,
            "num_return_sequences": 1,
        },
        model_names=models_selected,
        steps=checkpoints_selected,
    )


if __name__ == "__main__":
    # benchmark_pythia_models()
    benchmark_instruct_models()
