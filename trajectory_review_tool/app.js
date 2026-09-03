let map;
let layerGroup;
let currentPayload = null;

function speedColor(speed) {
    const value = Number(speed);
    if (!Number.isFinite(value)) return '#64748b';
    if (value === 0) return 'rgb(0, 0, 0)';
    if (value < 5) return 'rgb(26, 150, 65)';
    if (value < 10) return 'rgb(166, 217, 106)';
    if (value < 25) return 'rgb(203, 203, 15)';
    if (value < 80) return 'rgb(253, 174, 97)';
    if (value < 200) return 'rgb(215, 25, 28)';
    return 'rgb(129, 15, 124)';
}

function setStatus(message) {
    document.getElementById('status').textContent = message;
}

function setBusy(busy) {
    document.querySelectorAll('button').forEach(button => {
        button.disabled = busy;
    });
}

function observationLookup(observations) {
    const byId = {};
    observations.forEach(observation => {
        byId[String(observation.observation_id)] = observation;
    });
    return byId;
}

function transitionTooltip(transition) {
    const parts = [
        `<b>Transition ${transition.transition_id}</b>`,
        `Speed: ${Number(transition.speed_kmh).toFixed(2)} km/h`,
        `Distance: ${Number(transition.distance_m).toFixed(1)} m`,
        `Elapsed: ${Number(transition.elapsed_time_s).toFixed(0)} s`,
    ];
    if (transition.transition_plausibility !== 1) {
        parts.push(`<b>Rule violation:</b> ${transition.plausibility_reason || 'deterministic policy'}`);
    }
    return `<div class="transition-tooltip">${parts.join('<br>')}</div>`;
}

function renderMetadata(review) {
    const metadata = document.getElementById('metadata');
    metadata.innerHTML = `
        <dt>Dataset</dt><dd>${review.dataset}</dd>
        <dt>User</dt><dd>${review.user_id}</dd>
        <dt>Trajectory</dt><dd>${review.trajectory_id}</dd>
        <dt>Date</dt><dd>${review.start_date}</dd>
        <dt>Transitions</dt><dd>${review.n_transitions}</dd>
        <dt>Rules</dt><dd>Deterministic checks passed</dd>
    `;
}

function renderTrajectory(payload) {
    layerGroup.clearLayers();
    const review = payload.review;
    renderMetadata(review);
    const observations = observationLookup(payload.observations);
    const bounds = L.latLngBounds();
    const seenObservations = new Set();

    payload.transitions
        .sort((a, b) => Number(a.transition_order || 0) - Number(b.transition_order || 0))
        .forEach(transition => {
            const first = observations[String(transition.observation_id1)];
            const second = observations[String(transition.observation_id2)];
            if (!first || !second) return;
            [first, second].forEach(observation => {
                const id = String(observation.observation_id);
                if (seenObservations.has(id)) return;
                seenObservations.add(id);
                const marker = L.circleMarker([observation.lat, observation.lon], {
                    radius: 4,
                    color: '#1f2937',
                    weight: 1,
                    fillColor: 'white',
                    fillOpacity: 0.9,
                }).addTo(layerGroup);
                marker.bindTooltip(
                    `<b>Observation ${id}</b><br>${observation.timestamp || ''}<br>${Number(observation.lat).toFixed(5)}, ${Number(observation.lon).toFixed(5)}`
                );
                bounds.extend([observation.lat, observation.lon]);
            });
            const invalid = transition.transition_plausibility !== 1;
            const color = invalid ? '#dc2626' : speedColor(transition.speed_kmh);
            const line = L.polyline(
                [[first.lat, first.lon], [second.lat, second.lon]],
                {
                    color,
                    weight: invalid ? 6 : 4,
                    opacity: 0.9,
                    dashArray: invalid ? '6 6' : null,
                }
            ).addTo(layerGroup);
            line.bindTooltip(transitionTooltip(transition), { sticky: true });
        });

    if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [28, 28] });
    }
}

async function loadNext() {
    setBusy(true);
    setStatus('Loading a random unreviewed trajectory...');
    document.getElementById('notes').value = '';
    try {
        const response = await fetch('/api/next');
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Could not load trajectory');
        if (payload.done) {
            currentPayload = null;
            layerGroup.clearLayers();
            document.getElementById('metadata').innerHTML = '';
            setStatus('No unreviewed trajectory remains for the selected datasets.');
            return;
        }
        currentPayload = payload;
        renderTrajectory(payload);
        setStatus('Review this trajectory.');
    } catch (error) {
        setStatus(error.message);
    } finally {
        setBusy(false);
    }
}

async function submitReview(label) {
    if (!currentPayload) return;
    setBusy(true);
    setStatus(`Saving ${label} review...`);
    const review = currentPayload.review;
    try {
        const response = await fetch('/api/review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset: review.dataset,
                user_id: review.user_id,
                trajectory_id: review.trajectory_id,
                label,
                notes: document.getElementById('notes').value,
            }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Could not save review');
        await loadNext();
    } catch (error) {
        setStatus(error.message);
        setBusy(false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    map = L.map('map', { zoomControl: true }).setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);
    layerGroup = L.layerGroup().addTo(map);

    document.getElementById('label-unplausible').addEventListener('click', () => submitReview('unplausible'));
    document.getElementById('label-unsure').addEventListener('click', () => submitReview('unsure'));
    document.getElementById('label-plausible').addEventListener('click', () => submitReview('plausible'));
    loadNext();
});
