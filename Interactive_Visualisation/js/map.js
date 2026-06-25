// Read log again to find observations for that user on that specific date
async function updateMapForDate(user_id, dates) {
    try {
        highlightPlotlyDay(user_id, dates);

        currentMarkers.forEach(marker => leafletMap.removeLayer(marker));
        currentMarkers = [];

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

            const isUnplausible = t.is_unplausible === true;
            const isReviewedPlausible = t.reviewed_plausible === true;
            const polyline = L.polyline([[p1.lat, p1.lon], [p2.lat, p2.lon]], {
                color: isUnplausible ? '#ff4d4f' : (isReviewedPlausible ? '#00a676' : color),
                weight: isUnplausible || isReviewedPlausible ? 6 : 4,
                opacity: 0.9,
                dashArray: isUnplausible ? '5, 5' : (isReviewedPlausible ? '1, 8' : null)
            }).addTo(leafletMap);

            let transInfo = `<b>Speed:</b> ${parseFloat(t.speed).toFixed(2)} km/h<br><b>Distance:</b> ${parseFloat(t.distance).toFixed(2)} m<br><b>Elapsed Time:</b> ${parseFloat(t.elapsed_time).toFixed(0)} s<br><b>Plausibility:</b> ${parseFloat(t.transition_plausibility).toFixed(0)}`;
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
            if (t.reconstruction_error !== undefined) {
                transInfo += `<br><b>Model anomaly:</b> ${t.model_unplausible ? '<span style="color:red; font-weight:bold;">Yes (highest-error transition)</span>' : 'No'}`;
                transInfo += `<br><b>LSTM Error:</b> ${parseFloat(t.reconstruction_error).toFixed(4)}`;
                if (t.reconstruction_feature_errors && t.reconstruction_error > 0) {
                    const fErr = t.reconstruction_feature_errors;
                    transInfo += `<br><b>Feature Errors:</b>`;
                    Object.entries(fErr).forEach(([name, value]) => {
                        transInfo += `<br>&nbsp;&nbsp;${profileDisplayName(name)}: ${parseFloat(value).toFixed(4)}`;
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
