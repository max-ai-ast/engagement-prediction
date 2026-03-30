from utils.experiment_tracking import ClearMLExperimentTracker


class _FakeTask:
    def __init__(self) -> None:
        self.params = None

    def set_parameters_as_dict(self, params):
        self.params = params


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
