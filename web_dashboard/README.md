# Fast Local Dashboard

This is a lightweight local web dashboard that uses your existing cached `.pkl` files
from `data/dashboard_cache`. It does **not** recalculate anything; it just renders plots
on demand using the cached data and the existing Plotly helper functions.

## Run

From the repo root:

```
python web_dashboard/server.py
```

Then open:

```
http://127.0.0.1:8000
```

## Notes

- This server uses Python’s standard library HTTP server; no extra dependencies required.
- All plotting is done server‑side using your existing `utils/plotting_plotly.py`.
- UI updates only the plots that change, so it’s much faster than Streamlit.
