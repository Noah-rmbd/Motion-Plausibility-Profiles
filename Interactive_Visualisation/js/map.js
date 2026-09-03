function clearRouteMapLayers() {
    if (!leafletMap) return;
    currentMarkers.forEach(marker => leafletMap.removeLayer(marker));
    currentMarkers = [];
    if (window.routePolyline) {
        leafletMap.removeLayer(window.routePolyline);
        window.routePolyline = null;
    }
}

function clearSelectedTrajectoryLayers() {
    selectedTrajectoryLayers.forEach(layer => layer.remove());
    selectedTrajectoryLayers.clear();
    selectedTrajectoryIds.clear();
    hiddenSelectedTrajectoryIds.clear();
    selectedTrajectoryTimelineItems.clear();
    renderTimeline();
}

function trajectoryIdKey(value) {
    return String(value);
}

function selectedTrajectoryKey(dataset, trajectoryId) {
    return `${dataset}:${trajectoryIdKey(trajectoryId)}`;
}

function datasetShortLabel(dataset) {
    if (dataset === 'inat') return 'iNat';
    if (dataset === 'gowalla') return 'Gowalla';
    if (dataset === 'synthetic') return 'Synth';
    return dataset || '';
}

function clusterTrajectoryIds(cluster) {
    if (cluster.trajectory_ids) return cluster.trajectory_ids.map(trajectoryIdKey);
    if (cluster.trajectory_id !== undefined) return [trajectoryIdKey(cluster.trajectory_id)];
    return [];
}

function clusterIsFullySelected(cluster) {
    const dataset = currentDatasetName();
    const ids = clusterTrajectoryIds(cluster).map(id => selectedTrajectoryKey(dataset, id));
    return ids.length > 0 && ids.every(key => selectedTrajectoryIds.has(key));
}

function ensureTrajectoryBubbleLayer() {
    if (!leafletMap) return null;
    if (!trajectoryBubbleLayer) {
        trajectoryBubbleLayer = L.layerGroup().addTo(leafletMap);
    }
    return trajectoryBubbleLayer;
}

function clearTrajectoryBubbleLayer() {
    if (trajectoryBubbleLayer) {
        trajectoryBubbleLayer.clearLayers();
    }
    expandedBubbleLayers.forEach(item => {
        const layer = item.layer || item;
        if (layer && layer.remove) layer.remove();
    });
    expandedBubbleLayers = [];
}

function parentDescriptorString(parent) {
    return `${parent.zoom}:${parent.cell_size}:${parent.grid_x}:${parent.grid_y}`;
}

function formatBubbleCount(count) {
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`;
    if (count >= 1000) return `${(count / 1000).toFixed(count >= 10000 ? 0 : 1)}k`;
    return String(count);
}

function bubbleSize(count) {
    return Math.max(30, Math.min(76, 24 + Math.sqrt(count) * 4));
}

function updateBubbleControls(payload = null, message = '') {
    const controls = document.getElementById('bubble-controls');
    const status = document.getElementById('bubble-status');
    const backButton = document.getElementById('bubble-back-button');
    const resetButton = document.getElementById('bubble-reset-button');
    if (!controls || !status || !backButton || !resetButton) return;

    controls.hidden = false;
    backButton.disabled = expandedBubbleLayers.length === 0;
    resetButton.disabled = expandedBubbleLayers.length === 0;

    if (message) {
        status.textContent = message;
    } else if (payload) {
        const depth = expandedBubbleLayers.length;
        const depthLabel = depth > 0 ? ` Expanded bubbles ${depth}.` : '';
        const criterion = currentTrajectoryFilterCriterion();
        const thresholdLabel = appliedTrajectoryScoreFilter
            ? ` Score range ${formatModelScore(appliedTrajectoryScoreFilter.score_min)}-${formatModelScore(appliedTrajectoryScoreFilter.score_max)}.`
            : payload.threshold !== null && payload.threshold !== undefined
                ? ` Threshold ${formatModelScore(payload.threshold)}.`
                : '';
        const filterLabel = criterion === 'flight'
            ? `${payload.visible_points.toLocaleString()} plausible flight-speed (>200 km/h) trajectories, rendered as ${payload.clusters.length.toLocaleString()} bubbles/points.`
            : `${payload.visible_points.toLocaleString()} visible trajectory barycentres, rendered as ${payload.clusters.length.toLocaleString()} bubbles/points.`;
        status.textContent = `${filterLabel}${thresholdLabel}${depthLabel}`;
    } else {
        status.textContent = 'Choose a trajectory filter to render bubbles.';
    }
}

function trajectoryBubbleTooltip(cluster) {
    if (cluster.count > 1) {
        let content = `<b>${cluster.count.toLocaleString()} trajectories</b><br>${cluster.trajectory_ids ? 'Click to show these trajectories.' : 'Click to expand this bubble.'}`;
        if (cluster.score_summary) {
            content += `<br><b>Avg reconstruction error:</b> ${formatModelScore(cluster.score_summary.mean)}`;
            content += `<br><b>Score range:</b> ${formatModelScore(cluster.score_summary.min)} - ${formatModelScore(cluster.score_summary.max)}`;
        }
        return content;
    }
    const date = cluster.date || 'unknown date';
    let content = `<b>Trajectory ${cluster.trajectory_id}</b><br><b>User:</b> ${cluster.user_id}<br><b>Date:</b> ${date}<br><b>Transitions:</b> ${cluster.n_transitions}`;
    if (cluster.model_score !== undefined) {
        content += `<br><b>Model score:</b> ${formatModelScore(cluster.model_score)}`;
    }
    return content;
}

function observationLookupFromPayload(payload) {
    const observations = {};
    (payload.observations || []).forEach(obs => {
        observations[String(obs.observation_id)] = obs;
    });
    return observations;
}

function transitionScoreLookupFromPayload(payload) {
    const scores = {};
    (payload.transition_scores || []).forEach(score => {
        scores[String(score.transition_id)] = score;
    });
    return scores;
}

function trajectoryScoreLookupFromPayload(payload) {
    const scores = {};
    (payload.trajectory_scores || []).forEach(score => {
        scores[String(score.trajectory_id)] = score;
    });
    return scores;
}

function canonicalTransitionTooltip(transition, transitionScore = null, trajectoryScore = null) {
    const speed = Number(transition.speed_kmh ?? transition.speed);
    const distance = Number(transition.distance_m ?? transition.distance);
    const elapsed = Number(transition.elapsed_time_s ?? transition.elapsed_time);
    const acceleration = Number(transition.acceleration_m_s2 ?? transition.acceleration);
    const bearing = Number(transition.bearing_change_rad ?? transition.bearing_change);
    const baselineUnplausible = transition.transition_plausibility === 0;
    const dataset = transition.dataset || '';
    let transInfo = dataset
        ? `<b>Dataset:</b> ${datasetShortLabel(dataset)}<br><b>Trajectory:</b> ${transition.trajectory_id}<br><b>Transition:</b> ${transition.transition_id}`
        : `<b>Trajectory:</b> ${transition.trajectory_id}<br><b>Transition:</b> ${transition.transition_id}`;
    if (Number.isFinite(speed)) transInfo += `<br><b>Speed:</b> ${speed.toFixed(2)} km/h`;
    if (Number.isFinite(distance)) transInfo += `<br><b>Distance:</b> ${distance.toFixed(2)} m`;
    if (Number.isFinite(elapsed)) transInfo += `<br><b>Elapsed Time:</b> ${elapsed.toFixed(0)} s`;
    if (Number.isFinite(acceleration)) transInfo += `<br><b>Acceleration:</b> ${acceleration.toFixed(4)} m/s²`;
    if (Number.isFinite(bearing)) transInfo += `<br><b>Bearing Change:</b> ${bearing.toFixed(2)} rad`;
    transInfo += `<br><b>Deterministic baseline:</b> ${baselineUnplausible ? '<span style="color:red; font-weight:bold;">Violation</span>' : 'Valid'}`;
    if (baselineUnplausible && transition.plausibility_reason) {
        transInfo += `<br><b>Baseline reason:</b> ${transition.plausibility_reason}`;
    }
    const modelScore = transitionScore?.selected_score ?? transitionScore?.reconstruction_error ?? transitionScore?.mse ?? transitionScore?.score;
    if (modelScore !== undefined) {
        transInfo += `<br><b>Transition score:</b> ${formatModelScore(modelScore)}`;
    }
    const trajectoryModelScore = trajectoryScore?.selected_score ?? trajectoryScore?.model_score ?? trajectoryScore?.score;
    if (trajectoryModelScore !== undefined) {
        transInfo += `<br><b>Trajectory score:</b> ${formatModelScore(trajectoryModelScore)}`;
    }
    const featureEntries = Object.entries(transitionScore || {}).filter(([key]) => key.startsWith('error_'));
    if (featureEntries.length > 0) {
        transInfo += `<br><b>Feature Errors:</b>`;
        featureEntries.forEach(([name, value]) => {
            transInfo += `<br>&nbsp;&nbsp;${profileDisplayName(name.replace(/^error_/, ''))}: ${formatModelScore(value)}`;
        });
    }
    return transInfo;
}

function selectedObservationTooltip(observation) {
    const timestamp = observation.timestamp || observation.date || '';
    return `<b>ID:</b> ${observation.observation_id}<br><b>Lat:</b> ${Number(observation.lat).toFixed(5)}<br><b>Lon:</b> ${Number(observation.lon).toFixed(5)}<br><b>Date:</b> ${timestamp}`;
}

function selectedObservationIcon() {
    return L.icon({
        iconUrl: 'ressources/circle_icon.png',
        shadowUrl: 'ressources/circle_icon.png',
        iconSize: [20, 20],
        shadowSize: [0, 0],
        iconAnchor: [10, 10],
        shadowAnchor: [10, 10],
        popupAnchor: [0, -10]
    });
}

function renderSelectedTrajectoryPayload(payload) {
    const dataset = payload.dataset || currentDatasetName();
    const observations = observationLookupFromPayload(payload);
    const transitionScores = transitionScoreLookupFromPayload(payload);
    const trajectoryScores = trajectoryScoreLookupFromPayload(payload);
    const trajectoryMetadata = {};
    (payload.trajectories || []).forEach(trajectory => {
        trajectoryMetadata[trajectoryIdKey(trajectory.trajectory_id)] = trajectory;
    });
    const byTrajectory = {};
    (payload.transitions || []).forEach(transition => {
        byTrajectory[transition.trajectory_id] ||= [];
        byTrajectory[transition.trajectory_id].push(transition);
    });
    const bounds = L.latLngBounds();
    Object.entries(byTrajectory).forEach(([trajectoryId, transitions], index) => {
        const id = trajectoryIdKey(trajectoryId);
        const key = selectedTrajectoryKey(dataset, id);
        if (selectedTrajectoryLayers.has(key)) return;
        const trajectoryLayer = L.layerGroup().addTo(leafletMap);
        const addedObservationIds = new Set();
        const circleIcon = selectedObservationIcon();
        transitions
            .sort((a, b) => Number(a.transition_order || 0) - Number(b.transition_order || 0))
            .forEach(transition => {
                const first = observations[String(transition.observation_id1)];
                const second = observations[String(transition.observation_id2)];
                if (!first || !second) return;
                [first, second].forEach(observation => {
                    const observationId = String(observation.observation_id);
                    if (addedObservationIds.has(observationId)) return;
                    const marker = L.marker([observation.lat, observation.lon], { icon: circleIcon }).addTo(trajectoryLayer);
                    marker.bindTooltip(selectedObservationTooltip(observation));
                    addedObservationIds.add(observationId);
                });
                const transitionScore = transitionScores[String(transition.transition_id)];
                const trajectoryScore = trajectoryScores[id];
                transition.dataset = dataset;
                const speed = Number(transition.speed_kmh ?? transition.speed);
                const color = getSpeedColor(Number.isFinite(speed) ? `${speed} km/h` : '0 km/h');
                const baselineUnplausible = transition.transition_plausibility === 0 || transition.transition_plausibility === '0' || transition.baseline_unplausible === true || transition.is_unplausible === true;

                if (baselineUnplausible) {
                    // Create a multi-layered radial vanishing gradient (halo fading to transparent)
                    [20, 14, 9].forEach((w, idx) => {
                        L.polyline([[first.lat, first.lon], [second.lat, second.lon]], {
                            color: '#ffff00',
                            weight: w,
                            opacity: 0.15 * (idx + 1),
                            dashArray: null,
                            interactive: false,
                            className: 'unplausible-line-blurred'
                        }).addTo(trajectoryLayer);
                    });
                }
                const polyline = L.polyline([[first.lat, first.lon], [second.lat, second.lon]], {
                    color: baselineUnplausible ? '#000000' : color,
                    weight: baselineUnplausible ? 8 : 4,
                    opacity: 0.9,
                    dashArray: baselineUnplausible ? '5, 15' : null,
                    className: baselineUnplausible ? 'unplausible-line-blurred' : ''
                }).addTo(trajectoryLayer);
                const tooltip = canonicalTransitionTooltip(transition, transitionScore, trajectoryScore);
                polyline.bindTooltip(tooltip, { sticky: true });
                const p1Proj = leafletMap.project([first.lat, first.lon]);
                const p2Proj = leafletMap.project([second.lat, second.lon]);
                const angle = Math.atan2(p2Proj.y - p1Proj.y, p2Proj.x - p1Proj.x) * 180 / Math.PI;
                const midLat = (first.lat + second.lat) / 2;
                const midLon = (first.lon + second.lon) / 2;
                const arrowIcon = L.divIcon({
                    className: 'custom-arrow-icon',
                    html: `
                        <div style="width: 20px; height: 20px; transform: rotate(${angle}deg);">
                            <svg viewBox="0 0 24 24" width="20" height="20" style="overflow: visible;">
                                <polygon points="4,4 20,12 4,20" fill="${baselineUnplausible ? '#ffff00' : color}" stroke="${baselineUnplausible ? 'black' : 'white'}" stroke-width="2" />
                            </svg>
                        </div>
                    `,
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                });
                const arrowMarker = L.marker([midLat, midLon], { icon: arrowIcon, interactive: true }).addTo(trajectoryLayer);
                arrowMarker.bindTooltip(tooltip, { sticky: true });
                bounds.extend([first.lat, first.lon]);
                bounds.extend([second.lat, second.lon]);
            });
        if (trajectoryLayer.getLayers().length > 0) {
            selectedTrajectoryLayers.set(key, trajectoryLayer);
            selectedTrajectoryIds.add(key);
            hiddenSelectedTrajectoryIds.delete(key);
            const trajectory = trajectoryMetadata[id] || {};
            selectedTrajectoryTimelineItems.set(key, {
                key,
                dataset,
                trajectory_id: id,
                user_id: trajectory.user_id || transitions[0]?.user_id || '',
                date: trajectory.date || (trajectory.start_timestamp || '').slice(0, 10).replace(/-/g, '/'),
                n_transitions: trajectory.n_transitions ?? transitions.length,
                model_score: trajectoryScores[id]?.selected_score
                    ?? trajectoryScores[id]?.model_score
                    ?? trajectoryScores[id]?.score
                    ?? trajectory.model_score
            });
        } else {
            trajectoryLayer.remove();
        }
    });
    if (bounds.isValid()) {
        leafletMap.fitBounds(bounds, { padding: [50, 50] });
    }
    renderTimeline();
}

function showSelectedTrajectoryLayer(trajectoryKey) {
    const key = String(trajectoryKey);
    const layer = selectedTrajectoryLayers.get(key);
    if (!layer || !leafletMap) return false;
    if (!leafletMap.hasLayer(layer)) layer.addTo(leafletMap);
    selectedTrajectoryIds.add(key);
    hiddenSelectedTrajectoryIds.delete(key);
    renderTimeline();
    return true;
}

function hideSelectedTrajectoryLayer(trajectoryKey) {
    const key = String(trajectoryKey);
    const layer = selectedTrajectoryLayers.get(key);
    if (!layer || !leafletMap) return false;
    if (leafletMap.hasLayer(layer)) layer.remove();
    selectedTrajectoryIds.delete(key);
    hiddenSelectedTrajectoryIds.add(key);
    renderTimeline();
    updateBubbleControls(null, `Showing ${selectedTrajectoryIds.size.toLocaleString()} selected trajectories.`);
    return true;
}

function toggleSelectedTrajectoryLayer(trajectoryKey) {
    const key = String(trajectoryKey);
    if (selectedTrajectoryIds.has(key)) {
        return hideSelectedTrajectoryLayer(key);
    }
    return showSelectedTrajectoryLayer(key);
}

async function showClusterTrajectories(cluster, marker = null) {
    const dataset = currentDatasetName();
    const ids = clusterTrajectoryIds(cluster);
    ids
        .map(id => selectedTrajectoryKey(dataset, id))
        .filter(key => selectedTrajectoryLayers.has(key) && !selectedTrajectoryIds.has(key))
        .forEach(showSelectedTrajectoryLayer);
    const newIds = ids.filter(id => !selectedTrajectoryLayers.has(selectedTrajectoryKey(dataset, id)));
    if (newIds.length === 0) {
        if (marker) marker.remove();
        renderTimeline();
        return true;
    }
    const params = new URLSearchParams({
        dataset: currentDatasetName(),
        trajectory_ids: newIds.join(',')
    });
    if (currentModelSelection) params.set('model_id', currentModelSelection);
    const response = await fetch(`/api/trajectories?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
        updateBubbleControls(null, payload.error || 'Could not load trajectories for this bubble.');
        return true;
    }
    if (marker) marker.remove();
    renderSelectedTrajectoryPayload(payload);
    updateBubbleControls(null, `Showing ${selectedTrajectoryIds.size.toLocaleString()} selected trajectories.`);
    return true;
}

async function openTrajectoryFromBubble(cluster) {
    const userSelect = document.getElementById('user-select');
    if (userSelect) userSelect.value = cluster.user_id;
    await setUser(cluster.user_id);
    if (cluster.date) {
        selectedDate = [cluster.date];
        renderTimeline();
        await updateMapForDate(cluster.user_id, selectedDate);
    }
}

function clusterBounds(cluster) {
    if (!cluster.bounds) return null;
    const southWest = [cluster.bounds.south, cluster.bounds.west];
    const northEast = [cluster.bounds.north, cluster.bounds.east];
    return L.latLngBounds(southWest, northEast);
}

function currentTrajectoryFilterCriterion() {
    return document.getElementById('trajectory-filter-type')?.value || 'none';
}

function currentTrajectoryBubbleParams() {
    const criterion = currentTrajectoryFilterCriterion();
    const params = {};
    const modelSettings = currentScoreFilterSettings();
    if (modelSettings) {
        Object.assign(params, modelSettings);
    }
    if (criterion === 'user') {
        const selectedUser = document.getElementById('user-select')?.value || currentUser || '';
        if (!selectedUser) {
            return {
                error: 'Choose a user.'
            };
        }
        params.user_id = selectedUser;
    } else if (criterion === 'score') {
        if (!appliedTrajectoryScoreFilter) {
            return {
                error: 'Apply a score range.'
            };
        }
        Object.assign(params, appliedTrajectoryScoreFilter);
    } else if (criterion === 'flight') {
        params.min_transition_speed = '200';
        params.exclude_unplausible = 'true';
    } else if (criterion === 'none') {
        // No extra filtering: query trajectory bubbles for the whole dataset
    } else {
        return {
            error: 'Choose a trajectory filter to render bubbles.'
        };
    }
    return { params };
}

async function renderTrajectoryBubbles(options = {}) {
    if (!leafletMap) return;

    const layer = ensureTrajectoryBubbleLayer();
    if (!layer) return;
    const filter = currentTrajectoryBubbleParams();
    if (filter.error) {
        layer.clearLayers();
        updateBubbleControls(null, filter.error);
        return;
    }

    const bounds = leafletMap.getBounds();
    const clusterZoom = Math.round(leafletMap.getZoom());
    const params = new URLSearchParams({
        dataset: currentDatasetName(),
        zoom: String(clusterZoom),
        cell_size: '90',
        south: String(bounds.getSouth()),
        north: String(bounds.getNorth()),
        west: String(bounds.getWest()),
        east: String(bounds.getEast())
    });
    Object.entries(filter.params).forEach(([key, value]) => {
        params.set(key, String(value));
    });
    const activeOpenedIds = [...selectedTrajectoryIds]
        .filter(key => key.startsWith(`${currentDatasetName()}:`))
        .map(key => key.split(':')[1]);
    if (activeOpenedIds.length > 0) {
        params.set('exclude_trajectory_ids', activeOpenedIds.join(','));
    }
    const requestKey = params.toString();
    latestBubbleRequestKey = requestKey;

    updateBubbleControls(null, 'Loading trajectory bubbles...');

    try {
        const response = await fetch(`/api/trajectory_bubbles?${params.toString()}`);
        const payload = await response.json();
        if (latestBubbleRequestKey !== requestKey) return;
        if (!response.ok) {
            clearTrajectoryBubbleLayer();
            updateBubbleControls(null, payload.error || 'Trajectory bubbles are not available for this dataset.');
            return;
        }

        if (!options.preserveExpanded) {
            expandedBubbleLayers.forEach(item => {
                const layer = item.layer || item;
                if (layer && layer.remove) layer.remove();
            });
            expandedBubbleLayers = [];
        } else {
            const remaining = [];
            expandedBubbleLayers.forEach(item => {
                const parentZoom = item.parentZoom;
                const layer = item.layer || item;
                if (parentZoom !== undefined && clusterZoom <= parentZoom) {
                    if (layer && layer.remove) layer.remove();
                } else {
                    remaining.push(item);
                }
            });
            expandedBubbleLayers = remaining;
        }
        layer.clearLayers();
        renderClusterMarkers(payload.clusters, layer, clusterZoom);
        updateBubbleControls(payload);
    } catch (error) {
        if (latestBubbleRequestKey !== requestKey) return;
        clearTrajectoryBubbleLayer();
        updateBubbleControls(null, error.message || 'Could not load trajectory bubbles.');
    }
}

function renderClusterMarkers(clusters, layer, clusterZoom) {
    let minScore = Infinity;
    let maxScore = -Infinity;
    clusters.forEach(cluster => {
        const val = cluster.score_summary ? cluster.score_summary.mean : cluster.model_score;
        if (val !== undefined && val !== null && Number.isFinite(val)) {
            if (val < minScore) minScore = val;
            if (val > maxScore) maxScore = val;
        }
    });
    const hasScoreRange = Number.isFinite(minScore) && Number.isFinite(maxScore) && maxScore > minScore;

    clusters.forEach(cluster => {
        if (clusterIsFullySelected(cluster)) return;
        const isBubble = cluster.count > 1;
        const size = isBubble ? bubbleSize(cluster.count) : 13;
        const isBaselineImpossible = cluster.baseline_unplausible === true || cluster.is_unplausible === true;
        const isModelUnplausible = cluster.model_unplausible === true;
        let icon;
        if (isBubble) {
            let colorStyle = '';
            if (hasScoreRange && cluster.score_summary && Number.isFinite(cluster.score_summary.mean)) {
                const t = Math.max(0, Math.min(1, (cluster.score_summary.mean - minScore) / (maxScore - minScore)));
                const hue = Math.round((1 - t) * 120); // 120 = Green (lowest error), 0 = Red (highest error)
                colorStyle = `background: hsl(${hue}, 78%, 42%); color: #ffffff; border: 2px solid #ffffff; shadow: 0 2px 8px rgba(0,0,0,0.4);`;
            }
            icon = L.divIcon({
                className: '',
                html: `<div class="trajectory-bubble-marker" style="width:${size}px;height:${size}px;font-size:${Math.max(11, Math.min(16, size / 4))}px;${colorStyle}">${formatBubbleCount(cluster.count)}</div>`,
                iconSize: [size, size],
                iconAnchor: [size / 2, size / 2]
            });
        } else if (isBaselineImpossible) {
            icon = L.icon({
                iconUrl: 'ressources/impossible_circle_icon.png',
                iconSize: [20, 20],
                iconAnchor: [10, 10],
                popupAnchor: [0, -10]
            });
        } else {
            let pointStyle = '';
            if (isModelUnplausible) {
                pointStyle = 'background: #dc2626; border-color: #ffffff;';
            } else if (hasScoreRange && Number.isFinite(cluster.model_score)) {
                const t = Math.max(0, Math.min(1, (cluster.model_score - minScore) / (maxScore - minScore)));
                const hue = Math.round((1 - t) * 120);
                pointStyle = `background: hsl(${hue}, 78%, 42%);`;
            }
            icon = L.divIcon({
                className: '',
                html: `<div class="trajectory-point-marker" style="${pointStyle}"></div>`,
                iconSize: [size, size],
                iconAnchor: [size / 2, size / 2]
            });
        }

        const marker = L.marker([cluster.lat, cluster.lon], { icon }).addTo(layer);
        marker.bindTooltip(trajectoryBubbleTooltip(cluster), { sticky: true });
        marker.on('click', async event => {
            L.DomEvent.stopPropagation(event);
            if (clusterTrajectoryIds(cluster).length > 0) {
                await showClusterTrajectories(cluster, marker);
            } else if (cluster.count > 1) {
                await expandBubble(cluster, marker, clusterZoom);
            }
        });
    });
}

async function expandBubble(cluster, marker, currentClusterZoom) {
    const filter = currentTrajectoryBubbleParams();
    if (filter.error) {
        updateBubbleControls(null, filter.error);
        return;
    }
    const parent = parentDescriptorString(cluster.parent);
    const nextZoom = Math.min(24, currentClusterZoom + 2);
    const params = new URLSearchParams({
        dataset: currentDatasetName(),
        zoom: String(nextZoom),
        cell_size: '90',
        parents: parent
    });
    Object.entries(filter.params).forEach(([key, value]) => {
        params.set(key, String(value));
    });
    const activeOpenedIds = [...selectedTrajectoryIds]
        .filter(key => key.startsWith(`${currentDatasetName()}:`))
        .map(key => key.split(':')[1]);
    if (activeOpenedIds.length > 0) {
        params.set('exclude_trajectory_ids', activeOpenedIds.join(','));
    }
    try {
        const response = await fetch(`/api/trajectory_bubbles?${params.toString()}`);
        const payload = await response.json();
        if (!response.ok) {
            updateBubbleControls(null, payload.error || 'Could not expand this bubble.');
            return;
        }
        if (marker) marker.remove();
        const layer = L.layerGroup().addTo(leafletMap);
        expandedBubbleLayers.push({
            layer: layer,
            parentZoom: currentClusterZoom,
            expandedZoom: nextZoom
        });
        renderClusterMarkers(payload.clusters, layer, nextZoom);
        updateBubbleControls(null, `Expanded ${cluster.count.toLocaleString()} trajectories into ${payload.clusters.length.toLocaleString()} bubbles/points.`);
    } catch (error) {
        updateBubbleControls(null, error.message || 'Could not expand this bubble.');
    }
}

function resetTrajectoryBubbles() {
    expandedBubbleLayers.forEach(item => {
        const layer = item.layer || item;
        if (layer && layer.remove) layer.remove();
    });
    expandedBubbleLayers = [];
    updateBubbleControls();
}

function stepBackTrajectoryBubble() {
    const item = expandedBubbleLayers.pop();
    const layer = item ? (item.layer || item) : null;
    if (layer && layer.remove) layer.remove();
    updateBubbleControls();
}

function currentScoreFilterSettings() {
    if (!currentModelSelection) return null;
    const transitionScoreFeatures = selectedTransitionScoreFeatures();
    const params = {
        model_id: currentModelSelection,
        anomaly_percentage: document.getElementById('threshold-slider').value,
        aggregation: document.getElementById('aggregation-select').value,
        transition_score_mode: document.getElementById('transition-score-select').value,
        include_first_transition: document.getElementById('first-transition-select').value
    };
    if (transitionScoreFeatures?.length) {
        params.transition_score_features = transitionScoreFeatures.join(',');
    }
    return params;
}

function updateScoreFilterControls(message = '') {
    const status = document.getElementById('score-filter-status');
    const applyButton = document.getElementById('apply-score-filter');
    const hasModel = Boolean(currentModelSelection);
    if (applyButton) applyButton.disabled = !hasModel || !scoreDistributionPayload || modelConfigurationDirty;
    if (status) {
        status.textContent = message
            || (modelConfigurationDirty ? 'Save model to update score filtering.' : hasModel ? '' : 'No model selected.');
    }
}

function renderScoreDistributionSelection() {
    if (!scoreDistributionPayload || !window.Plotly) return;
    const minValue = Number(document.getElementById('score-range-min').value);
    const maxValue = Number(document.getElementById('score-range-max').value);
    const shapes = [];
    if (Number.isFinite(minValue) && Number.isFinite(maxValue)) {
        shapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: minValue,
            x1: maxValue,
            y0: 0,
            y1: 1,
            fillcolor: 'rgba(39, 111, 191, 0.18)',
            line: { width: 0 },
            layer: 'below'
        });
    }
    Plotly.relayout('score-distribution-plot', { shapes });
    updateSelectedScoreRangeSummary();
}

function setScoreRangeControlBounds(payload) {
    const min = Number(payload.min);
    const max = Number(payload.max);
    const span = Math.max(max - min, 0.0001);
    const step = Math.max(span / 1000, 0.0001);
    ['score-range-min', 'score-range-max', 'score-range-min-slider', 'score-range-max-slider'].forEach(id => {
        const input = document.getElementById(id);
        if (!input) return;
        input.min = String(min);
        input.max = String(min + span);
        input.step = String(step);
    });
}

function setScoreRangeValues(minValue, maxValue) {
    const min = Number(scoreDistributionPayload?.min ?? minValue);
    const max = Number(scoreDistributionPayload?.max ?? maxValue);
    let lower = Math.max(min, Math.min(max, Number(minValue)));
    let upper = Math.max(min, Math.min(max, Number(maxValue)));
    if (lower > upper) {
        [lower, upper] = [upper, lower];
    }
    document.getElementById('score-range-min').value = lower.toFixed(6);
    document.getElementById('score-range-max').value = upper.toFixed(6);
    document.getElementById('score-range-min-slider').value = String(lower);
    document.getElementById('score-range-max-slider').value = String(upper);
    scoreRangeSelection = { min: lower, max: upper };
    renderScoreDistributionSelection();
}

function syncScoreRangeFromSliders(event) {
    const minSlider = document.getElementById('score-range-min-slider');
    const maxSlider = document.getElementById('score-range-max-slider');
    let lower = Number(minSlider.value);
    let upper = Number(maxSlider.value);
    if (lower > upper) {
        if (event?.target === minSlider) {
            upper = lower;
        } else {
            lower = upper;
        }
    }
    setScoreRangeValues(lower, upper);
}

function syncScoreRangeFromInputs() {
    setScoreRangeValues(
        Number(document.getElementById('score-range-min').value),
        Number(document.getElementById('score-range-max').value)
    );
}

function selectedScoreRangeApproximation() {
    if (!scoreDistributionPayload) return null;
    const minValue = Number(document.getElementById('score-range-min').value);
    const maxValue = Number(document.getElementById('score-range-max').value);
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue > maxValue) return null;
    let selected = 0;
    scoreDistributionPayload.histogram.forEach(bin => {
        const left = Number(bin.bin_left);
        const right = Number(bin.bin_right);
        if (right <= left || right < minValue || left > maxValue) return;
        const overlap = Math.max(0, Math.min(right, maxValue) - Math.max(left, minValue));
        selected += Number(bin.count || 0) * (overlap / (right - left));
    });
    return {
        count: Math.round(selected),
        percentage: scoreDistributionPayload.total_points > 0
            ? selected / scoreDistributionPayload.total_points * 100
            : 0
    };
}

function updateSelectedScoreRangeSummary(prefix = '') {
    const status = document.getElementById('score-filter-status');
    if (!status) return;
    const estimate = selectedScoreRangeApproximation();
    if (!estimate) {
        status.textContent = prefix || (currentModelSelection ? 'Select a valid score range.' : 'No model selected.');
        return;
    }
    const lead = prefix ? `${prefix} ` : '';
    status.textContent = `${lead}${estimate.count.toLocaleString()} trajectories in range (${estimate.percentage.toFixed(1)}% of iNaturalist + Gowalla).`;
}

function drawScoreDistribution(payload) {
    if (!window.Plotly) return;
    const x = payload.histogram.map(bin => (bin.bin_left + bin.bin_right) / 2);
    const y = payload.histogram.map(bin => bin.count);
    const widths = payload.histogram.map(bin => bin.bin_right - bin.bin_left);
    Plotly.newPlot('score-distribution-plot', [{
        type: 'bar',
        x,
        y,
        width: widths,
        marker: { color: '#276fbf' },
        hovertemplate: 'Score %{x:.4f}<br>Trajectories %{y}<extra></extra>'
    }], {
        margin: { l: 34, r: 8, t: 8, b: 30 },
        height: 155,
        dragmode: false,
        xaxis: { title: '', fixedrange: true },
        yaxis: { title: '', fixedrange: true },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    }, { responsive: true, displayModeBar: false });
}

function restoreScoreDistributionView() {
    if (!scoreDistributionPayload || !window.Plotly) return;
    drawScoreDistribution(scoreDistributionPayload);
    if (scoreRangeSelection) {
        setScoreRangeValues(scoreRangeSelection.min, scoreRangeSelection.max);
    } else {
        renderScoreDistributionSelection();
    }
    setTimeout(() => {
        Plotly.Plots.resize('score-distribution-plot');
    }, 0);
}

async function loadScoreDistribution() {
    const settings = currentScoreFilterSettings();
    if (!settings) {
        updateScoreFilterControls('No model selected.');
        return;
    }
    updateScoreFilterControls('Loading score distribution...');
    const params = new URLSearchParams({
        dataset: 'combined_real',
        ...settings
    });
    const requestKey = params.toString();
    latestScoreDistributionRequestKey = requestKey;
    const previousRange = scoreDistributionSettingsKey === requestKey
        ? scoreRangeSelection
        : null;
    try {
        const response = await fetch(`/api/trajectory_score_distribution?${params.toString()}`);
        const payload = await response.json();
        if (latestScoreDistributionRequestKey !== requestKey) return;
        if (!response.ok) {
            scoreDistributionPayload = null;
            updateScoreFilterControls(payload.error || 'Could not load score distribution.');
            return;
        }
        scoreDistributionPayload = payload;
        scoreDistributionSettingsKey = requestKey;
        const lower = previousRange
            ? previousRange.min
            : Number.isFinite(payload.threshold)
                ? payload.threshold
                : payload.min;
        const upper = previousRange ? previousRange.max : payload.max;
        setScoreRangeControlBounds(payload);
        drawScoreDistribution(payload);
        setScoreRangeValues(Number(lower), Number(upper));
        updateScoreFilterControls();
        updateSelectedScoreRangeSummary(`${payload.count.toLocaleString()} scored trajectories loaded.`);
    } catch (error) {
        if (latestScoreDistributionRequestKey !== requestKey) return;
        scoreDistributionPayload = null;
        scoreDistributionSettingsKey = '';
        updateScoreFilterControls(error.message || 'Could not load score distribution.');
    }
}

function scheduleScoreDistributionRefresh() {
    if (scoreDistributionTimer) clearTimeout(scoreDistributionTimer);
    if (modelConfigurationDirty) {
        updateScoreFilterControls('Save model to update score filtering.');
        return;
    }
    if (currentTrajectoryFilterCriterion() !== 'score') {
        updateScoreFilterControls();
        return;
    }
    if (!currentModelSelection) {
        scoreDistributionPayload = null;
        scoreDistributionSettingsKey = '';
        scoreRangeSelection = null;
        appliedTrajectoryScoreFilter = null;
        updateScoreFilterControls('No model selected.');
        return;
    }
    updateScoreFilterControls('Updating score distribution...');
    scoreDistributionTimer = setTimeout(loadScoreDistribution, 250);
}

function applyScoreFilterToMap() {
    const settings = currentScoreFilterSettings();
    const minScore = Number(document.getElementById('score-range-min').value);
    const maxScore = Number(document.getElementById('score-range-max').value);
    if (!settings) {
        updateScoreFilterControls('No model selected.');
        return;
    }
    if (!Number.isFinite(minScore) || !Number.isFinite(maxScore) || minScore > maxScore) {
        updateScoreFilterControls('Invalid score range.');
        return;
    }
    appliedTrajectoryScoreFilter = {
        ...settings,
        score_min: minScore,
        score_max: maxScore
    };
    const filterType = document.getElementById('trajectory-filter-type');
    if (filterType) filterType.value = 'score';
    refreshTrajectoryBubblesForFilter();
}

function syncTrajectoryFilterControls() {
    const criterion = currentTrajectoryFilterCriterion();
    const userSelect = document.getElementById('user-select');
    const userControls = document.getElementById('user-filter-controls');
    const scoreControls = document.getElementById('score-filter-controls');
    const timeline = document.getElementById('timeline-container');
    if (userControls) userControls.hidden = criterion !== 'user';
    if (scoreControls) scoreControls.hidden = criterion !== 'score';
    if (userSelect) userSelect.disabled = criterion !== 'user';
    if (timeline) timeline.hidden = false;
    renderTimeline();
    updateScoreFilterControls();
}

function refreshTrajectoryBubblesForFilter({ resetDrill = true, clearRoute = true } = {}) {
    syncTrajectoryFilterControls();
    if (resetDrill) {
        expandedBubbleLayers.forEach(layer => layer.remove());
        expandedBubbleLayers = [];
    }
    if (clearRoute) clearRouteMapLayers();
    renderTrajectoryBubbles();
}

function initializeTrajectoryBubbleControls() {
    const filterType = document.getElementById('trajectory-filter-type');
    if (filterType) {
        filterType.addEventListener('change', () => {
            syncTrajectoryFilterControls();
            scheduleScoreDistributionRefresh();
            refreshTrajectoryBubblesForFilter();
        });
    }
    const backButton = document.getElementById('bubble-back-button');
    const resetButton = document.getElementById('bubble-reset-button');
    if (backButton) backButton.addEventListener('click', stepBackTrajectoryBubble);
    if (resetButton) resetButton.addEventListener('click', resetTrajectoryBubbles);
    const applyScoreButton = document.getElementById('apply-score-filter');
    if (applyScoreButton) applyScoreButton.addEventListener('click', applyScoreFilterToMap);
    ['score-range-min', 'score-range-max'].forEach(id => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('input', syncScoreRangeFromInputs);
            input.addEventListener('change', syncScoreRangeFromInputs);
        }
    });
    ['score-range-min-slider', 'score-range-max-slider'].forEach(id => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('input', syncScoreRangeFromSliders);
            input.addEventListener('change', syncScoreRangeFromSliders);
        }
    });
    syncTrajectoryFilterControls();
    updateBubbleControls();
}

// Read log again to find observations for that user on that specific date
async function updateMapForDate(user_id, dates) {
    try {
        highlightPlotlyDay(user_id, dates);

        clearRouteMapLayers();

        var circleIcon = L.icon({
            iconUrl: 'ressources/circle_icon.png',
            shadowUrl: 'ressources/circle_icon.png',
            iconSize: [20, 20],
            shadowSize: [0, 0],
            iconAnchor: [10, 10],
            shadowAnchor: [10, 10],
            popupAnchor: [0, -10]
        });

        const isAnimated = document.getElementById('animate-path') && document.getElementById('animate-path').checked;
        let bounds = L.latLngBounds();

        const userTransitions = appData.transitions_list.filter(t => String(t.user_id) === String(user_id) && dates.includes(t.date.trim()));
        const userObscured = appData.obscured_observations.filter(o => String(o.user_id) === String(user_id) && dates.includes(o.date.trim()));
        const userObservations = appData.observations.filter(o => String(o.user_id) === String(user_id) && dates.includes(o.date.trim().substring(0, 10).replace(/-/g, '/')));

        let transitionObsIds = new Set();
        for (let t of userTransitions) {
            transitionObsIds.add(String(t.observation_id1).replace('iN-p', '').replace('iN-o', ''));
            transitionObsIds.add(String(t.observation_id2).replace('iN-p', '').replace('iN-o', ''));
        }
        let isolatedObs = userObservations.filter(o => !transitionObsIds.has(String(o.observation_id)));

        if (userTransitions.length === 0 && userObscured.length === 0 && isolatedObs.length === 0) {
            console.log("No valid points to display for this date.");
            return;
        }

        // Draw unobscured points
        let addedPoints = new Set();
        for (let t of userTransitions) {
            let id1 = String(t.observation_id1).replace('iN-p', '').replace('iN-o', '');
            let id2 = String(t.observation_id2).replace('iN-p', '').replace('iN-o', '');
            let p1 = appData.obsCoordsMap[id1];
            let p2 = appData.obsCoordsMap[id2];

            if (p1 && !addedPoints.has(t.observation_id1)) {
                let time1 = appData.obsTimestamps ? (appData.obsTimestamps[t.observation_id1] || "") : "";
                if (time1) time1 = ` ${time1}`;
                const marker = L.marker([p1.lat, p1.lon], { icon: circleIcon }).addTo(leafletMap);
                marker.bindTooltip(`<b>ID:</b> ${t.observation_id1}<br><b>Lat:</b> ${p1.lat.toFixed(5)}<br><b>Lon:</b> ${p1.lon.toFixed(5)}<br><b>Date:</b> ${t.date}${time1}`);
                // Highlight the corresponding profile shape on hover
                const obsKey1 = 'obs_' + String(t.observation_id1);
                marker.on('mouseover', () => { if (shapeIndexMap[obsKey1] !== undefined) highlightProfileShape(shapeIndexMap[obsKey1]); });
                marker.on('mouseout', clearProfileHighlight);
                currentMarkers.push(marker);
                bounds.extend([p1.lat, p1.lon]);
                addedPoints.add(t.observation_id1);
            }
            if (p2 && !addedPoints.has(t.observation_id2)) {
                let time2 = appData.obsTimestamps ? (appData.obsTimestamps[t.observation_id2] || "") : "";
                if (time2) time2 = ` ${time2}`;
                const marker = L.marker([p2.lat, p2.lon], { icon: circleIcon }).addTo(leafletMap);
                marker.bindTooltip(`<b>ID:</b> ${t.observation_id2}<br><b>Lat:</b> ${p2.lat.toFixed(5)}<br><b>Lon:</b> ${p2.lon.toFixed(5)}<br><b>Date:</b> ${t.date}${time2}`);
                const obsKey2 = 'obs_' + String(t.observation_id2);
                marker.on('mouseover', () => { if (shapeIndexMap[obsKey2] !== undefined) highlightProfileShape(shapeIndexMap[obsKey2]); });
                marker.on('mouseout', clearProfileHighlight);
                currentMarkers.push(marker);
                bounds.extend([p2.lat, p2.lon]);
                addedPoints.add(t.observation_id2);
            }
        }

        // Draw isolated unobscured points
        for (let o of isolatedObs) {
            let p1 = { lat: o.lat, lon: o.long };
            if (!addedPoints.has(String(o.observation_id))) {
                let timeO = appData.obsTimestamps ? (appData.obsTimestamps[o.observation_id] || "") : "";
                if (timeO) timeO = ` ${timeO}`;
                const marker = L.marker([p1.lat, p1.lon], { icon: circleIcon }).addTo(leafletMap);
                marker.bindTooltip(`<b>ID:</b> ${o.observation_id}<br><b>Lat:</b> ${p1.lat.toFixed(5)}<br><b>Lon:</b> ${p1.lon.toFixed(5)}<br><b>Date:</b> ${o.date}${timeO}`);
                const obsKey = 'obs_' + String(o.observation_id);
                marker.on('mouseover', () => { if (shapeIndexMap[obsKey] !== undefined) highlightProfileShape(shapeIndexMap[obsKey]); });
                marker.on('mouseout', clearProfileHighlight);
                currentMarkers.push(marker);
                bounds.extend([p1.lat, p1.lon]);
                addedPoints.add(String(o.observation_id));
            }
        }

        console.log(`Currently displaying ${addedPoints.size} observations on the map.`);

        // Obscured points are intentionally NOT drawn on the map because they lack precise coordinates.

        leafletMap.fitBounds(bounds, { padding: [50, 50] });
        if (isAnimated) await sleep(800);

        for (let t of userTransitions) {
            let id1 = String(t.observation_id1).replace('iN-p', '').replace('iN-o', '');
            let id2 = String(t.observation_id2).replace('iN-p', '').replace('iN-o', '');
            let p1 = appData.obsCoordsMap[id1];
            let p2 = appData.obsCoordsMap[id2];
            if (!p1 || !p2) continue;

            let color = getSpeedColor(t.speed + " km/h");

            const isUnplausible = t.is_unplausible === true || t.baseline_unplausible === true || t.transition_plausibility === 0 || t.transition_plausibility === '0';
            const isReviewedPlausible = t.reviewed_plausible === true;

            if (isUnplausible) {
                // Create a multi-layered radial vanishing gradient (halo fading to transparent)
                [20, 14, 9].forEach((w, idx) => {
                    L.polyline([[p1.lat, p1.lon], [p2.lat, p2.lon]], {
                        color: '#ffff00',
                        weight: w,
                        opacity: 0.15 * (idx + 1),
                        dashArray: null,
                        interactive: false,
                        className: 'unplausible-line-blurred'
                    }).addTo(leafletMap);
                });
            }

            const polyline = L.polyline([[p1.lat, p1.lon], [p2.lat, p2.lon]], {
                color: isUnplausible ? '#000000' : (isReviewedPlausible ? '#00a676' : color),
                weight: isUnplausible || isReviewedPlausible ? 8 : 4,
                opacity: 0.9,
                dashArray: isUnplausible ? '5, 15' : (isReviewedPlausible ? '1, 8' : null),
                className: isUnplausible ? 'unplausible-line-blurred' : ''
            }).addTo(leafletMap);

            let transInfo = `<b>Trajectory:</b> ${t.trajectory_id}<br><b>Transition:</b> ${t.transition_id}<br><b>Speed:</b> ${parseFloat(t.speed).toFixed(2)} km/h<br><b>Distance:</b> ${parseFloat(t.distance).toFixed(2)} m<br><b>Elapsed Time:</b> ${parseFloat(t.elapsed_time).toFixed(0)} s<br><b>Plausibility:</b> ${parseFloat(t.transition_plausibility).toFixed(0)}`;
            if (t.acceleration !== undefined) {
                transInfo += `<br><b>Acceleration:</b> ${parseFloat(t.acceleration).toFixed(4)} m/s²`;
            }
            if (t.bearing_change !== undefined) {
                transInfo += `<br><b>Bearing Change:</b> ${parseFloat(t.bearing_change).toFixed(2)}rad`;
            }
            transInfo += `<br><b>Deterministic baseline:</b> ${t.baseline_unplausible ? '<span style="color:red; font-weight:bold;">Violation</span>' : 'Valid'}`;
            if (t.baseline_unplausible && t.plausibility_reason) {
                transInfo += `<br><b>Baseline reason:</b> ${t.plausibility_reason}`;
            }
            transInfo += `<br><b>Reviewed plausible:</b> ${t.reviewed_plausible ? '<span style="color:#00a676; font-weight:bold;">Yes</span>' : 'No'}`;
            if (t.reconstruction_error !== undefined || t.model_score !== undefined) {
                transInfo += `<br><b>Model anomaly:</b> ${t.model_unplausible ? '<span style="color:red; font-weight:bold;">Yes (highest-error transition)</span>' : 'No'}`;
                transInfo += `<br><b>Transition score:</b> ${formatModelScore(t.model_score ?? t.reconstruction_error)}`;
                if (t.reconstruction_feature_errors && Object.keys(t.reconstruction_feature_errors).length > 0) {
                    const fErr = t.reconstruction_feature_errors;
                    transInfo += `<br><b>Feature-wise scores:</b>`;
                    Object.entries(fErr).forEach(([name, value]) => {
                        transInfo += `<br>&nbsp;&nbsp;${profileDisplayName(name)}: ${formatModelScore(value)}`;
                    });
                }
            }
            // sticky: true makes the tooltip follow the mouse along the polyline
            polyline.bindTooltip(transInfo, { sticky: true });

            // Highlight the corresponding profile shape on hover
            const transKey = 'trans_' + String(t.transition_id);
            polyline.on('mouseover', () => { if (shapeIndexMap[transKey] !== undefined) highlightProfileShape(shapeIndexMap[transKey]); });
            polyline.on('mouseout', clearProfileHighlight);
            currentMarkers.push(polyline);

            const p1_proj = leafletMap.project([p1.lat, p1.lon]);
            const p2_proj = leafletMap.project([p2.lat, p2.lon]);
            const angle = Math.atan2(p2_proj.y - p1_proj.y, p2_proj.x - p1_proj.x) * 180 / Math.PI;
            const midLat = (p1.lat + p2.lat) / 2;
            const midLon = (p1.lon + p2.lon) / 2;

            const arrowIcon = L.divIcon({
                className: 'custom-arrow-icon',
                html: `
                    <div style="width: 20px; height: 20px; transform: rotate(${angle}deg);">
                        <svg viewBox="0 0 24 24" width="20" height="20" style="overflow: visible;">
                            <polygon points="4,4 20,12 4,20" fill="${isUnplausible ? '#ff4d4f' : color}" stroke="${isReviewedPlausible && !isUnplausible ? '#00a676' : 'white'}" stroke-width="2" />
                        </svg>
                    </div>
                `,
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            });

            const arrowMarker = L.marker([midLat, midLon], { icon: arrowIcon, interactive: true }).addTo(leafletMap);
            arrowMarker.bindTooltip(transInfo, { sticky: true });
            arrowMarker.on('mouseover', () => { if (shapeIndexMap[transKey] !== undefined) highlightProfileShape(shapeIndexMap[transKey]); });
            arrowMarker.on('mouseout', clearProfileHighlight);
            currentMarkers.push(arrowMarker);

            if (isAnimated) await sleep(400);
        }

    } catch (error) {
        console.error('Error updating map:', error);
    }
}
