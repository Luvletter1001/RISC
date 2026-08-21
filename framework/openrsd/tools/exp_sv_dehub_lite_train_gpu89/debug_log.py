"""NDJSON debug logger for session cd5945."""
from __future__ import annotations

import json
import time
from pathlib import Path

_LOG = Path('/data1/zcy/OpenRSD/.cursor/debug-cd5945.log')
_SESSION = 'cd5945'


def dbg(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = 'pre-fix') -> None:
    # #region agent log
    try:
        payload = {
            'sessionId': _SESSION,
            'runId': run_id,
            'hypothesisId': hypothesis_id,
            'location': location,
            'message': message,
            'data': data,
            'timestamp': int(time.time() * 1000),
        }
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
    except Exception:
        pass
    # #endregion
