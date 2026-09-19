"""Phase telemetry shared by initial and refreshed ANCE rounds."""
import json
import time
from pathlib import Path


class PhaseTimer:
    def __init__(self, events=None):
        self.timings = {}
        self.events = Path(events) if events else None
        self._name = None
        self._t0 = None

    def _emit(self, event):
        event['at_epoch'] = time.time()
        print(f'[Mining] {json.dumps(event, sort_keys=True)}', flush=True)
        if self.events:
            self.events.parent.mkdir(parents=True, exist_ok=True)
            with self.events.open('a') as handle:
                handle.write(json.dumps(event) + '\n')

    def __call__(self, name):
        self.close()
        self._name, self._t0 = name, time.monotonic()
        self._emit({'phase': name, 'event': 'start'})
        return self

    def close(self):
        if self._name is not None:
            elapsed = time.monotonic() - self._t0
            self.timings[f'{self._name}_s'] = elapsed
            self._emit({'phase': self._name, 'event': 'done', 'seconds': elapsed})
            self._name = None

    def fail(self, error):
        self._emit({'phase': self._name, 'event': 'failed', 'error': repr(error)})
        self.close()

    def merge_encoder(self, prefix, pickle_path):
        path = Path(str(pickle_path) + '.timings.json')
        if path.exists():
            data = json.loads(path.read_text())
            for name, value in data.items():
                self.timings[f'{prefix}_{name}'] = value
