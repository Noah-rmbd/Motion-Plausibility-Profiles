async function fetchModelScoresForUser(user_id) {
    if (!currentModelSelection) return;

    const userTransitions = appData.transitions_list.filter(t => String(t.user_id) === String(user_id));
    const transIds = userTransitions.map(t => t.transition_id);
    
    const userTrajectories = appData.trajectories_list.filter(t => String(t.user_id) === String(user_id));
    const trajIds = userTrajectories.map(t => t.trajectory_id);
    
    if (transIds.length === 0 && trajIds.length === 0) return;
    
    const anomalyPercentage = parseFloat(document.getElementById('threshold-slider').value);
    const aggregation = document.getElementById('aggregation-select').value;
    const transitionScoreMode = document.getElementById('transition-score-select').value;
    const transitionScoreFeatures = selectedTransitionScoreFeatures();
    const includeFirstTransition =
        document.getElementById('first-transition-select').value === 'true';
    const requestKey = JSON.stringify({
        user_id: String(user_id),
        model_id: currentModelSelection,
        dataset: currentDatasetName(),
        anomaly_percentage: anomalyPercentage,
        aggregation,
        transition_score_mode: transitionScoreMode,
        transition_score_features: transitionScoreFeatures || [],
        include_first_transition: includeFirstTransition
    });
    latestScoreRequestKey = requestKey;
    
    try {
        const response = await fetch('/api/scores', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: currentModelSelection,
                dataset: currentDatasetName(),
                anomaly_percentage: anomalyPercentage,
                aggregation,
                transition_score_mode: transitionScoreMode,
                transition_score_features: transitionScoreFeatures,
                include_first_transition: includeFirstTransition,
                transition_ids: transIds,
                trajectory_ids: trajIds
            })
        });
        
        const data = await response.json();
        if (latestScoreRequestKey !== requestKey) return;
        if (!response.ok) {
            document.getElementById('threshold-reference').textContent =
                data.error || 'Could not update model scores for this feature selection.';
            return;
        }
        currentThreshold = data.threshold;
        document.getElementById('threshold-value').textContent = `Threshold: ${formatModelScore(currentThreshold)}`;
        const featureLabel = data.transition_score_features?.length
            ? ` · features: ${data.transition_score_features.join(', ')}`
            : '';
        document.getElementById('threshold-reference').textContent =
            `Reference: ${data.threshold_reference || 'current dataset'}${featureLabel}`;
            
        userTransitions.forEach(t => {
            const newScores = data.transitions[String(t.transition_id)];
            if (newScores) {
                t.model_unplausible = newScores.is_unplausible;
                t.is_unplausible = t.baseline_unplausible || t.model_unplausible;
                t.model_score = Number(newScores.selected_score ?? newScores.mse ?? newScores.score);
                t.reconstruction_error = t.model_score;
                const featureMap = {};
                (newScores.feature_names || []).forEach((name, index) => {
                    featureMap[name] = newScores.features[index];
                });
                t.reconstruction_feature_errors = featureMap;
            }
        });
            
        userTrajectories.forEach(t => {
            const newScores = data.trajectories[String(t.trajectory_id)];
            if (newScores) {
                t.model_unplausible = newScores.is_unplausible;
                t.is_unplausible = t.baseline_unplausible || t.model_unplausible;
                t.model_score = Number(newScores.selected_score ?? newScores.model_score ?? newScores.score);
            }
        });
    } catch (e) {
        console.error("Failed to fetch model scores", e);
    }
}

async function fetchAndParseUsers() {
    try {
        const prefix = currentDatasetName();
        const response = await fetch(`/api/users?dataset=${prefix}`);
        appData.users_list = await response.json();

        appData.transitions_list = [];
        appData.trajectories_list = [];
        appData.observations = [];
        appData.obsCoordsMap = {};
        appData.obscured_observations = [];
        appData.obsTimestamps = {};

        const users = appData.users_list.map(u => String(u.username));
        console.log(`Found ${users.length} users via API.`);
        return users;
    } catch (error) {
        console.error('Error fetching API users:', error);
        return [];
    }
}

// Set the map, the timeline and load the pdf files
async function setUser(user_id) {
    selectedDate = [];
    currentUser = user_id;
    syncMppEmptyState();
    console.log(`Great news ${user_id}`);
    
    const prefix = currentDatasetName();
    const response = await fetch(`/api/user_data?dataset=${prefix}&user_id=${user_id}`);
    const data = await response.json();
    
    appData.observations = data.observations || [];
    appData.transitions_list = data.transitions || [];
    appData.trajectories_list = data.trajectories || [];
    appData.obscured_observations = data.obscured_observations || [];
    
    appData.obsCoordsMap = {};
    for (let obs of appData.observations) {
        appData.obsCoordsMap[obs.observation_id] = { lat: obs.lat, lon: obs.lon };
        appData.obsCoordsMap[String(obs.observation_id).replace('iN-p', '').replace('iN-o', '')] = { lat: obs.lat, lon: obs.lon };
        appData.obsTimestamps[obs.observation_id] = String(obs.date || '').substring(11);
    }
    
    appData.transitions_list.forEach(t => {
        t.baseline_unplausible = t.baseline_unplausible === true ||
            (t.transition_plausibility !== undefined && t.transition_plausibility === 0);
        t.reviewed_plausible = t.reviewed_plausible === true;
        t.model_unplausible = false;
        t.is_unplausible = t.baseline_unplausible;
    });
    
    appData.trajectories_list.forEach(t => {
        t.baseline_unplausible = t.baseline_unplausible === true;
        t.reviewed_plausible = t.reviewed_plausible === true;
        t.model_unplausible = false;
        t.is_unplausible = t.baseline_unplausible;
    });

    await fetchModelScoresForUser(user_id);
    
    await setTimeline(user_id);
    renderPlotlyProfile(user_id);
    if (globalDaysArray.length > 0) {
        selectedDate = [globalDaysArray[0]];
        renderTimeline();
    }
    refreshTrajectoryBubblesForFilter({ clearRoute: true });
    updatePlausibleLabelAction();
}

function selectedTrajectoryIdsForPlausibleReview() {
    if (!currentUser || currentDatasetName() === 'synthetic') return [];
    const selectedDays = new Set(selectedDate.map(day => String(day).trim()));
    if (selectedDays.size === 0) return [];
    const ids = appData.trajectories_list
        .filter(traj =>
            String(traj.user_id) === String(currentUser)
            && selectedDays.has(String(traj.date || '').trim())
            && Number(traj.n_transitions || 0) > 2
            && !traj.baseline_unplausible
        )
        .map(traj => Number(traj.trajectory_id));
    return [...new Set(ids)].filter(Number.isFinite);
}

function updatePlausibleLabelAction(message = '') {
    const button = document.getElementById('mark-plausible-button');
    const status = document.getElementById('plausible-label-status');
    if (!button) return;
    const ids = selectedTrajectoryIdsForPlausibleReview();
    const synthetic = currentDatasetName() === 'synthetic';
    button.disabled = !currentUser || synthetic || ids.length === 0;
    button.textContent = ids.length > 0 ? `Mark plausible (${ids.length})` : 'Mark plausible';
    if (status && message) {
        status.textContent = message;
    } else if (status && synthetic) {
        status.textContent = 'Synthetic trajectories cannot be labeled plausible.';
    } else if (status && currentUser && selectedDate.length > 0 && ids.length === 0) {
        status.textContent = 'No eligible trajectory in the selected timeline period.';
    } else if (status && !message) {
        status.textContent = '';
    }
}

async function markSelectedTrajectoriesPlausible() {
    const ids = selectedTrajectoryIdsForPlausibleReview();
    if (!currentUser || ids.length === 0) {
        updatePlausibleLabelAction('Select at least one eligible real trajectory.');
        return;
    }
    const confirmed = window.confirm(
        `Store ${ids.length} selected trajectory/trajectories as manually reviewed plausible?`
    );
    if (!confirmed) return;
    const response = await fetch('/api/plausible_trajectories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            dataset: currentDatasetName(),
            user_id: currentUser,
            trajectory_ids: ids
        })
    });
    const payload = await response.json();
    if (!response.ok) {
        updatePlausibleLabelAction(payload.error || 'Could not save plausible labels.');
        return;
    }
    const added = payload.added?.length || 0;
    const existing = payload.existing?.length || 0;
    const rejected = payload.rejected?.length || 0;
    let message = `${added} added, ${existing} already stored`;
    if (rejected > 0) message += `, ${rejected} rejected`;
    const addedIds = new Set((payload.added || []).map(row => Number(row.trajectory_id)));
    const existingIds = new Set((payload.existing || []).map(Number));
    const reviewedIds = new Set([...addedIds, ...existingIds]);
    if (reviewedIds.size > 0) {
        appData.trajectories_list.forEach(traj => {
            if (reviewedIds.has(Number(traj.trajectory_id))) {
                traj.reviewed_plausible = true;
            }
        });
        appData.transitions_list.forEach(transition => {
            if (reviewedIds.has(Number(transition.trajectory_id))) {
                transition.reviewed_plausible = true;
            }
        });
        renderTimeline();
        renderPlotlyProfile(currentUser);
        if (selectedDate.length > 0) {
            updateMapForDate(currentUser, selectedDate);
        }
    }
    updatePlausibleLabelAction(message);
    if (rejected > 0) {
        console.warn('Rejected plausible labels:', payload.rejected);
    }
}
