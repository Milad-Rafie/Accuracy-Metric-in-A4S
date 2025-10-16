import uuid

import uuid
import numpy as np
import pandas as pd
from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.metric_registries.model_metric_registry import ModelMetric
from a4s_eval.service.model_functional import FunctionalModel
from a4s_eval.service.model_load import load_model
import pytest

import numpy as np
from pathlib import Path


from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Model,
    ModelConfig,
    ModelFramework,
)

from tests.save_measures_utils import save_measures


@pytest.fixture
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )

    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(test_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = test_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(train_data, data_shape: DataShape) -> Dataset:
    data = train_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


@pytest.fixture
def functional_model() -> FunctionalModel:
    model_config = ModelConfig(
        path="./tests/data/lcld_v2_tabtransformer.pt", framework=ModelFramework.TORCH
    )
    return load_model(model_config)


def test_non_empty_registry():
    assert len(model_metric_registry._functions) > 0


@pytest.mark.parametrize("evaluator_function", model_metric_registry)
def test_data_metric_registry_contains_evaluator(
    evaluator_function: tuple[str, ModelMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: FunctionalModel,
):
    measures = evaluator_function[1](
        data_shape, ref_model, test_dataset, functional_model
    )
    save_measures(evaluator_function[0], measures)
    assert len(measures) > 0


def test_accuracy_batch_10000(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: FunctionalModel,
):
    
    N_BATCHES = 5
    BATCH_SIZE = 10_000
    total_rows = N_BATCHES * BATCH_SIZE

    base = test_dataset.data.copy()
    reps = int(np.ceil(total_rows / len(base)))
    big = pd.concat([base] * reps, ignore_index=True).iloc[:total_rows]

    out_csv = Path("tests/data/measures/accuracy.csv")
    if out_csv.exists():
        out_csv.unlink()

    accuracy_fn = model_metric_registry._functions["accuracy"]

    all_measures = []
    for i in range(N_BATCHES):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_df = big.iloc[start:end].copy()
        test_dataset.data = batch_df

        measures = accuracy_fn(data_shape, ref_model, test_dataset, functional_model)
        assert measures, f"No measures returned for batch {i}"
        # append measures from this batch (usually length 1)
        all_measures.extend(measures)

    save_measures("accuracy", all_measures)

    assert out_csv.exists(), "accuracy.csv was not created"
    with out_csv.open() as f:
        lines = sum(1 for _ in f)
    assert lines >= N_BATCHES + 1, f"Expected at least {N_BATCHES} datapoints, found {lines-1}"