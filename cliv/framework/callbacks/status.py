import lightning as L


class StatusCallback(L.Callback):
    def __init__(self, status: dict):
        self.status = status
        self.exception_triggered = False

    def on_exception(self, trainer, pl_module, exception):
        self.exception_triggered = True

    def teardown(self, trainer, pl_module, stage):
        if not self.exception_triggered:
            self.status[stage] = True
