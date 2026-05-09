"""MLflow-backed model registry wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class ModelRegistry:
    def __init__(self, tracking_uri: str | None = None) -> None:
        import mlflow
        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def log_run(
        self,
        experiment: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifact_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> str:
        self.mlflow.set_experiment(experiment)
        with self.mlflow.start_run(run_name=run_name) as run:
            self.mlflow.log_params(params)
            self.mlflow.log_metrics(metrics)
            if artifact_dir:
                self.mlflow.log_artifacts(str(artifact_dir))
            return run.info.run_id

    def register(self, model_uri: str, name: str, stage: str = "Staging") -> None:
        from mlflow import register_model
        from mlflow.client import MlflowClient
        mv = register_model(model_uri=model_uri, name=name)
        MlflowClient().transition_model_version_stage(
            name=name, version=mv.version, stage=stage, archive_existing_versions=True,
        )
