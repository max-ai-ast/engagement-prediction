from utils.experiment_tracking import ClearMLExperimentTracker


class _FakeTask:
    def __init__(self) -> None:
        self.params = None
        self.artifacts = []

    def set_parameters_as_dict(self, params):
        self.params = params

    def upload_artifact(self, name, artifact_object):
        self.artifacts.append((name, artifact_object))
        return "artifact-id"


def test_log_params_updates_clearml_parameters_with_section_prefix():
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._task = _FakeTask()

    tracker.log_params(
        params={
            "run_dir": "/tmp/run",
            "run_name": "20260320_123456_all",
        },
        name="Directories",
    )

    assert tracker._task.params == {
        "Directories/run_dir": "/tmp/run",
        "Directories/run_name": "20260320_123456_all",
    }


def test_log_file_artifact_uploads_path_to_clearml_task(tmp_path):
    artifact_path = tmp_path / "author_idx.parquet"
    artifact_path.write_bytes(b"parquet")
    tracker = ClearMLExperimentTracker.__new__(ClearMLExperimentTracker)
    tracker._task = _FakeTask()

    result = tracker.log_file_artifact("author_idx_mapping", artifact_path)

    assert result == "artifact-id"
    assert tracker._task.artifacts == [("author_idx_mapping", str(artifact_path))]
