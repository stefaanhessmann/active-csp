import os.path
from typing import Optional, Dict
import yaml


class StateTracker:

    def __init__(
        self,
        state_path: str,
        initial_state: Optional[Dict] = None,
    ):
        self.state_path = state_path
        if initial_state is None:
            initial_state = dict()

        if os.path.exists(state_path):
            self._load_state()
        else:
            self.set_state(initial_state)

    def update_state(self, **kwargs):
        state = self.get_state()
        state.update(**kwargs)
        self.set_state(state)

    def get_state(self, key: Optional[str] = None):
        if key is not None:
            return self.state[key]
        return self.state

    def set_state(self, state):
        with open(self.state_path, "w") as file:
            yaml.dump(state, file)
        self.state = state

    def _load_state(self):
        with open(self.state_path, "rb") as file:
            state = yaml.full_load(file)
        self.state = state
