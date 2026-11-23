# QSML 0.3 — Possibility Engine

QSML is a quantum-inspired software project for classical computers. It simulates
quantum state vectors and demonstrates amplitude amplification (Grover search).
It does **not** convert physical transistors into qubits or provide genuine
quantum-hardware speedup.

## Your development machine
Designed around the reported HP Victus setup: Windows 11, Intel Core i5,
16 GB RAM, Intel UHD Graphics and NVIDIA GeForce RTX 2050.

## Open in VS Code
1. Extract this ZIP.
2. In VS Code, choose **File → Open Folder** and select `qsml_v0_3`.
3. Open **Terminal → New Terminal**.
4. Run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m qsml
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Run tests:
```powershell
python -m unittest discover -s tests -v
```

## What's new in 0.3
- Possibility Engine using a state-vector representation.
- Grover amplitude amplification for small unstructured search examples.
- Candidate ranking by measured probability.
- Memory guard before allocating a state vector.
- Resource planner now accounts for working-memory overhead.

## Try the engine
The default demo searches 3-bit strings and marks `101`:
```powershell
python -m qsml
```

Try a custom search from Python:
```python
from qsml import grover_search

result = grover_search(
    num_qubits=3,
    marked_states=["001", "110"],
    shots=1000,
    seed=42,
)
print(result.top_candidates)
print(result.measurements)
```

Bit strings are written most-significant bit first in user-facing output.
Internally, qubit 0 is the least-significant bit.

## Memory safety
A state vector requires 16 * 2^n bytes for complex128 amplitudes. Operations may
need additional temporary memory. QSML checks available RAM before allocation
and uses a conservative working-memory allowance. The check is a guardrail, not
a guarantee: close memory-heavy applications before large simulations.

## Files
- `qsml/register.py`: state-vector and quantum gates
- `qsml/possibility.py`: Grover search / amplitude amplification
- `qsml/hardware.py`: CPU, RAM, storage and GPU inspection
- `qsml/resources.py`: memory estimates and safe qubit planning
- `qsml/__main__.py`: runnable demonstration
- `tests/`: automated tests

## Roadmap
- 0.4: circuit files and a small QSML circuit language.
- 0.5: optimization problem interface and classical baselines.
- Later: benchmark optional GPU acceleration against CPU before adopting it.
