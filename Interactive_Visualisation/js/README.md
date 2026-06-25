# Interactive Visualization JavaScript

The visualization uses classic browser scripts loaded in dependency order from
`main.html`. Files share the same browser global scope; keep cross-file state in
`state.js` and keep feature logic in the file matching the UI area it serves.

- `state.js`: shared application state and dataset helpers.
- `utils.js`: formatting, speed colors, and small shared helpers.
- `stats.js`: dataset/model population statistics panel.
- `models.js`: model list, model filters, scoring controls, and metric panels.
- `data.js`: API calls for users, scores, user trajectories, and manual labels.
- `profile.js`: Plotly motion profile rendering and highlighting.
- `map.js`: Leaflet trajectory rendering.
- `timeline.js`: timeline grouping, selection, and anomaly/plausible markers.
- `app.js`: startup, event wiring, map legend, panels, and resizer behavior.
