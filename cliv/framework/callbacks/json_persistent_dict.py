import json
import os


class DictPersistJSON(dict):
    def __init__(self, filename, *args, **kwargs):
        self.filename = filename
        self._load()
        self.update(*args, **kwargs)

    def _load(self):
        if os.path.isfile(self.filename):
            with open(self.filename, "r") as fh:
                self.update(json.load(fh))

    def _dump(self):
        with open(self.filename, "w") as fh:
            json.dump(self, fh)

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        else:
            return None

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        self._dump()

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
        self._dump()
