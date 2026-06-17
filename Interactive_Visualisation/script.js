let appData = {
    users_list: [],
    transitions_list: [],
    trajectories_list: [],
    obscured_observations: [],
    observations: [],
    obsCoordsMap: {}
};

let currentDatabase = 'original';

let currentModelSelection = '';
let allAvailableModels = [];
let currentThreshold = Infinity;
let originalTransitionsData = {};
let originalTrajectoriesData = {};
let currentStatsPayload = null;
const visibleDatasets = new Set(['inat', 'gowalla', 'synthetic']);

function currentDatasetName() {
    return currentDatabase === 'original' ? 'inat' : currentDatabase;
}

function modelUsesVisibleDatasetsOnly(model) {
    const trainingDatasets = model.training?.datasets || [];
    const evaluatedDatasets = model.datasets?.map(dataset => dataset.dataset) || [];
    return trainingDatasets.every(dataset => visibleDatasets.has(dataset))
        && evaluatedDatasets.every(dataset => visibleDatasets.has(dataset));
}

const statsMetricMetadata = {
    speed: { label: 'Speed', unit: 'km/h' },
    acceleration: { label: 'Acceleration', unit: 'm/s²' },
    distance: { label: 'Distance', unit: 'm' },
    elapsed_time: { label: 'Elapsed time', unit: 's' },
    bearing_change: { label: 'Bearing change', unit: 'rad' },
    trajectory_n_transitions: { label: 'Transitions per trajectory', unit: 'transitions' },
    max_speed: { label: 'Maximum trajectory speed', unit: 'km/h' },
    total_distance: { label: 'Total trajectory distance', unit: 'm' },
    total_elapsed_time: { label: 'Total trajectory elapsed time', unit: 's' },
    model_score: { label: 'Model anomaly score', unit: 'score' }
};

function formatStatNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '--';
    if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function downloadStatsFile(filename, content, type) {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([content], { type }));
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
}

async function loadDatasetStats() {
    const dataset = document.getElementById('stats-dataset').value;
    const subset = document.getElementById('stats-subset').value;
    const metric = document.getElementById('stats-metric').value;
    const isModelPopulation = [
        'model_flagged',
        'model_accepted',
        'transition_flagged',
        'transition_accepted'
    ].includes(subset);
    const isTransitionPopulation = subset === 'transition_flagged' || subset === 'transition_accepted';
    if (isTransitionPopulation && ['trajectory_n_transitions', 'max_speed', 'total_distance', 'total_elapsed_time'].includes(metric)) {
        document.getElementById('stats-metric').value = 'speed';
        return loadDatasetStats();
    }
    let response;
    if (isModelPopulation) {
        if (!currentModelSelection) {
            throw new Error('Select a model before loading model population statistics.');
        }
        response = await fetch('/api/model_stats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: currentModelSelection,
                dataset,
                population: subset,
                metric,
                anomaly_percentage: parseFloat(
                    document.getElementById('threshold-slider').value
                ),
                aggregation: document.getElementById('aggregation-select').value,
                transition_score_mode:
                    document.getElementById('transition-score-select').value,
                include_first_transition:
                    document.getElementById('first-transition-select').value === 'true'
            })
        });
    } else {
        if (['max_speed', 'total_distance', 'total_elapsed_time', 'model_score'].includes(metric)) {
            document.getElementById('stats-metric').value = 'speed';
            return loadDatasetStats();
        }
        response = await fetch(`/api/stats?dataset=${dataset}&subset=${subset}&metric=${metric}`);
    }
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Statistics request failed');
    }
    currentStatsPayload = await response.json();
    renderDatasetStats(currentStatsPayload);
}

function renderDatasetStats(payload) {
    const metadata = statsMetricMetadata[payload.metric];
    const summary = payload.summary;
    const quantiles = summary.quantiles || {};
    let modelContext = '';
    if (payload.model_id) {
        if (Number.isFinite(Number(payload.flagged_transitions))) {
            modelContext = `
                <dt>Flagged transitions</dt><dd>${payload.flagged_transitions.toLocaleString()} (${formatMetric(payload.flagged_transition_rate)})</dd>
                <dt>Threshold</dt><dd>${formatModelScore(payload.threshold)} transition score</dd>`;
        } else {
            modelContext = `
                <dt>Flagged trajectories</dt><dd>${payload.flagged_trajectories.toLocaleString()} (${formatMetric(payload.flagged_rate)})</dd>
                <dt>Threshold</dt><dd>${formatModelScore(payload.threshold)} trajectory score</dd>`;
        }
    }
    document.getElementById('stats-summary').innerHTML = `
        <dl>
            <dt>Trajectories</dt><dd>${payload.n_trajectories.toLocaleString()}</dd>
            <dt>Transitions</dt><dd>${payload.n_transitions.toLocaleString()}</dd>
            <dt>Mean</dt><dd>${formatStatNumber(summary.mean)} ${metadata.unit}</dd>
            <dt>Median</dt><dd>${formatStatNumber(quantiles.p50)} ${metadata.unit}</dd>
            <dt>P95</dt><dd>${formatStatNumber(quantiles.p95)} ${metadata.unit}</dd>
            <dt>P99</dt><dd>${formatStatNumber(quantiles.p99)} ${metadata.unit}</dd>
            ${modelContext}
        </dl>`;

    const traces = [{
        type: 'bar',
        name: payload.subset === 'model_flagged' ? 'Model flagged' : (
            payload.subset === 'model_accepted' ? 'Model accepted' : (
                payload.subset === 'transition_flagged' ? 'Flagged transitions' : (
                    payload.subset === 'transition_accepted' ? 'Accepted transitions' : 'Selected population'
                )
            )
        ),
        x: payload.histogram.map(bin => (bin.bin_left + bin.bin_right) / 2),
        y: payload.histogram.map(bin => bin.count),
        width: payload.histogram.map(bin => bin.bin_right - bin.bin_left),
        marker: { color: '#287d5b' },
        hovertemplate: `%{x:.4g} ${metadata.unit}<br>%{y:,} rows<extra></extra>`
    }];
    if (payload.comparison_histogram && payload.comparison_histogram.length) {
        traces.unshift({
            type: 'bar',
            name: payload.comparison_label,
            x: payload.comparison_histogram.map(bin => (bin.bin_left + bin.bin_right) / 2),
            y: payload.comparison_histogram.map(bin => bin.count),
            width: payload.comparison_histogram.map(bin => bin.bin_right - bin.bin_left),
            marker: { color: '#9aa2a0', opacity: 0.48 },
            hovertemplate: `%{x:.4g} ${metadata.unit}<br>%{y:,} rows<extra></extra>`
        });
    }
    Plotly.react('stats-plot', traces, {
        margin: { l: 58, r: 18, t: 34, b: 52 },
        title: { text: `${metadata.label} distribution`, font: { size: 14 } },
        xaxis: { title: metadata.unit, fixedrange: false },
        yaxis: { title: 'Count', fixedrange: false },
        bargap: 0.02,
        barmode: 'overlay',
        paper_bgcolor: 'white',
        plot_bgcolor: '#f7f8f7'
    }, {
        responsive: true,
        displaylogo: false
    });

    const featureErrors = payload.feature_errors || [];
    document.getElementById('stats-feature-errors').innerHTML = featureErrors.length
        ? `
            <h3>Feature reconstruction errors</h3>
            <table class="metric-table">
                <thead><tr><th>Feature</th><th>Mean</th><th>vs all</th></tr></thead>
                <tbody>
                    ${featureErrors.slice(0, 8).map(row => `
                        <tr>
                            <td>${profileDisplayName(row.feature)}</td>
                            <td>${formatModelScore(row.mean_error)}</td>
                            <td>${Number.isFinite(row.lift) ? `${row.lift.toFixed(2)}×` : '--'}</td>
                        </tr>`).join('')}
                </tbody>
            </table>`
        : '';

    const quality = payload.data_quality || {};
    document.getElementById('stats-quality').innerHTML = Object.keys(quality).length
        ? `
            <h3>Data quality</h3>
            <dl>
                <dt>Undefined acceleration</dt><dd>${Number(quality.undefined_acceleration || 0).toLocaleString()}</dd>
                <dt>Undefined bearing</dt><dd>${Number(quality.undefined_bearing_change || 0).toLocaleString()}</dd>
                <dt>Stationary with turn</dt><dd>${Number(quality.stationary_nonzero_bearing_change || 0).toLocaleString()}</dd>
                <dt>Zero elapsed time</dt><dd>${Number(quality.zero_elapsed_time || 0).toLocaleString()}</dd>
            </dl>`
        : '';
}

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models');
        if (response.ok) {
            const models = await response.json();
            allAvailableModels = models.filter(modelUsesVisibleDatasetsOnly);
        }
    } catch (e) {
        console.error("Failed to fetch available models", e);
    }
    updateModelDropdown();
}

async function loadModelMetrics(modelId) {
    if (!modelId) return;
    const model = allAvailableModels.find(item => item.model_id === modelId);
    if (!model || model.synthetic_metrics) return;
    const response = await fetch(
        `/api/model_metrics?model_id=${encodeURIComponent(modelId)}`
    );
    if (!response.ok) return;
    model.synthetic_metrics = await response.json();
    if (currentModelSelection === modelId) {
        updateModelInformation();
        updatePerformanceMetrics();
    }
}

function updateModelDropdown() {
    const modelSelect = document.getElementById('model-select');
    if (!modelSelect) return;
    
    const dbPrefix = currentDatasetName();
    
    modelSelect.innerHTML = '';
    const filters = {
        model_type: document.getElementById('filter-model-type')?.value || '',
        training_mode: document.getElementById('filter-training-mode')?.value || '',
        feature_representation: document.getElementById('filter-feature-representation')?.value || ''
    };
    const relevantModels = allAvailableModels.filter(m =>
        m.datasets.some(dataset => dataset.dataset === dbPrefix)
        && (!filters.model_type || m.model_type === filters.model_type)
        && (!filters.training_mode || modelTrainingMode(m) === filters.training_mode)
        && (!filters.feature_representation || modelFeatureRepresentation(m) === filters.feature_representation)
    );
    relevantModels.forEach(m => {
        const option = document.createElement('option');
        option.value = m.model_id;
        option.textContent = `${m.model_type.replace(/_/g, ' ')} · ${m.features.length} features · ${m.training.epochs} epochs`;
        modelSelect.appendChild(option);
    });
    currentModelSelection = modelSelect.value;
    updateModelInformation();
    updatePerformanceMetrics();
    loadModelMetrics(currentModelSelection);
}

function modelTrainingPopulation(model) {
    return [...model.training.datasets].sort().join('+');
}

function modelTrainingMode(model) {
    const population = modelTrainingPopulation(model);
    if (population === 'inat') return 'inat';
    if (population === 'gowalla') return 'gowalla';
    return (model.training.sampling_strategy || 'all') === 'balanced'
        ? 'combined_weighted'
        : 'combined';
}

function modelFeatureRepresentation(model) {
    if (model.experiment && model.experiment.feature_representation) {
        return model.experiment.feature_representation;
    }
    return model.features.some(feature => feature.startsWith('speed_'))
        ? 'mpp_speed_ranges'
        : 'continuous_speed';
}

function formatPipelineValue(value) {
    const labels = {
        'all': 'All trajectories',
        'balanced': 'Balanced datasets',
        'continuous_speed': 'Continuous speed',
        'mpp_speed_ranges': 'MPP speed ranges',
        'distance_time_bearing': 'Distance, time, bearing',
        'mpp_distance_time_bearing': 'MPP ranges, distance, time, bearing',
        'mpp_speed_accel_bearing_distance_time': 'MPP speed, acceleration, distance, time, bearing',
        'mpp_speed_accel_bearing_distance_time_speed': 'MPP speed, speed, acceleration, distance, time, bearing',
        'mpp_speed_accel_distance_time_speed': 'MPP speed, speed, acceleration, distance, time',
        'mpp_speed_bearing_distance_time_speed': 'MPP speed, speed, distance, time, bearing',
        'mean_no_bearing': 'Feature mean, no bearing',
        'weighted_no_bearing': 'Concept weighted, no bearing',
        'mean_no_acceleration': 'Feature mean, no acceleration',
        'weighted_no_acceleration': 'Concept weighted, no acceleration',
        'combined': 'iNaturalist + Gowalla',
        'combined_weighted': 'iNaturalist + Gowalla, balanced',
        'false': 'Exclude transition 0',
        'true': 'Include transition 0',
        'gowalla+inat': 'iNaturalist + Gowalla',
        'inat': 'iNaturalist only',
        'gowalla': 'Gowalla only'
    };
    return labels[value] || value.replace(/_/g, ' ');
}

function populateModelFilters() {
    const definitions = [
        ['filter-model-type', model => model.model_type],
        ['filter-training-mode', modelTrainingMode],
        ['filter-feature-representation', modelFeatureRepresentation]
    ];
    definitions.forEach(([id, getter]) => {
        const select = document.getElementById(id);
        if (!select) return;
        const previous = select.value;
        const values = [...new Set(allAvailableModels.map(getter))].sort();
        select.innerHTML = '<option value="">All</option>';
        values.forEach(value => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = formatPipelineValue(value);
            select.appendChild(option);
        });
        if (values.includes(previous)) select.value = previous;
    });
}

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
    const includeFirstTransition =
        document.getElementById('first-transition-select').value === 'true';
    
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
                include_first_transition: includeFirstTransition,
                transition_ids: transIds,
                trajectory_ids: trajIds
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            currentThreshold = data.threshold;
            document.getElementById('threshold-value').textContent = `Threshold: ${formatModelScore(currentThreshold)}`;
            document.getElementById('threshold-reference').textContent =
                `Reference: ${data.threshold_reference || 'current dataset'}`;
            
            userTransitions.forEach(t => {
                if (originalTransitionsData[t.transition_id] === undefined) {
                    originalTransitionsData[t.transition_id] = {
                        is_unplausible: t.is_unplausible,
                        mse: t.reconstruction_error,
                        features: t.reconstruction_feature_errors,
                        plausibility_reason: t.plausibility_reason
                    };
                }
                const newScores = data.transitions[t.transition_id];
                if (newScores) {
                    t.model_unplausible = newScores.is_unplausible;
                    t.is_unplausible = t.baseline_unplausible || t.model_unplausible;
                    t.reconstruction_error = newScores.mse;
                    const featureMap = {};
                    (newScores.feature_names || []).forEach((name, index) => {
                        featureMap[name] = newScores.features[index];
                    });
                    t.reconstruction_feature_errors = featureMap;
                }
            });
            
            userTrajectories.forEach(t => {
                if (originalTrajectoriesData[t.trajectory_id] === undefined) {
                    originalTrajectoriesData[t.trajectory_id] = t.is_unplausible;
                }
                const newScores = data.trajectories[t.trajectory_id];
                if (newScores) {
                    t.model_unplausible = newScores.is_unplausible;
                    t.is_unplausible = t.baseline_unplausible || t.model_unplausible;
                }
            });
        }
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

// Store the currently selected user
let currentUser = null;
let selectedDate = [];
let leafletMap = null;
let currentMarkers = [];
let isDraggingTimeline = false;
let globalDaysArray = [];

document.addEventListener('mouseup', () => {
    if (isDraggingTimeline) {
        isDraggingTimeline = false;
        if (currentUser) {
            updateMapForDate(currentUser, selectedDate);
        }
    }
});

// Set the map, the timeline and load the pdf files
async function setUser(user_id) {
    selectedDate = [];
    currentUser = user_id;
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
    
    originalTransitionsData = {};
    originalTrajectoriesData = {};
    
    appData.transitions_list.forEach(t => {
        t.baseline_unplausible = t.baseline_unplausible === true ||
            (t.transition_plausibility !== undefined && t.transition_plausibility === 0);
        t.model_unplausible = false;
        t.is_unplausible = t.baseline_unplausible;
        originalTransitionsData[t.transition_id] = {
            is_unplausible: t.is_unplausible,
            mse: 0.0,
            features: [0,0,0,0,0,0],
            plausibility_reason: t.plausibility_reason
        };
    });
    
    appData.trajectories_list.forEach(t => {
        t.baseline_unplausible = t.baseline_unplausible === true;
        t.model_unplausible = false;
        t.is_unplausible = t.baseline_unplausible;
        originalTrajectoriesData[t.trajectory_id] = t.is_unplausible;
    });

    await fetchModelScoresForUser(user_id);
    
    await setTimeline(user_id);
    renderPlotlyProfile(user_id);
    if (globalDaysArray.length > 0) {
        selectedDate = [globalDaysArray[0]];
        renderTimeline();
        await updateMapForDate(user_id, selectedDate);
    }
}


let isColorblindMode = false;

const colorPaletteStandard = {
    c0: 'rgb(0, 0, 0)',
    c5: 'rgb(26, 150, 65)',
    c10: 'rgb(166, 217, 106)',
    c25: 'rgb(203, 203, 15)',
    c80: 'rgb(253, 174, 97)',
    c200: 'rgb(215, 25, 28)',
    cMax: 'rgb(129, 15, 124)'
};

const colorPaletteColorblind = {
    c0: 'rgb(0, 0, 0)',
    c5: 'rgb(27, 120, 55)',
    c10: 'rgb(127, 191, 123)',
    c25: 'rgb(217, 240, 211)',
    c80: 'rgb(231, 212, 232)',
    c200: 'rgb(175, 141, 195)',
    cMax: 'rgb(118, 42, 131)'
};

// Function to get color from speed
function getSpeedColor(speedStr) {
    if (!speedStr) return 'gray';
    const numMatch = speedStr.match(/[\d.]+/);
    if (!numMatch) return 'gray';
    const speed = parseFloat(numMatch[0]);

    const p = isColorblindMode ? colorPaletteColorblind : colorPaletteStandard;

    if (speed === 0) return p.c0;
    if (speed < 5) return p.c5;
    if (speed < 10) return p.c10;
    if (speed < 25) return p.c25;
    if (speed < 80) return p.c80;
    if (speed < 200) return p.c200;
    return p.cMax;
}

// Function for async sleep
const sleep = ms => new Promise(r => setTimeout(r, ms));

function formatModelScore(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '--';
    return number >= 0.01 ? number.toFixed(4) : number.toExponential(2);
}

function usesNativeTrajectoryScore(model) {
    return Boolean(model?.architecture?.trajectory_score);
}

function updateModelScoringControls(model) {
    const nativeScore = usesNativeTrajectoryScore(model);
    const controls = [
        ['transition-score-select', 'mean'],
        ['aggregation-select', 'mean'],
        ['first-transition-select', 'true'],
    ];
    controls.forEach(([id, nativeDefault]) => {
        const control = document.getElementById(id);
        if (!control) return;
        if (nativeScore) control.value = nativeDefault;
        control.disabled = nativeScore;
    });
}

function updateModelInformation() {
    const model = allAvailableModels.find(item => item.model_id === currentModelSelection);
    const content = document.getElementById('model-info-content');
    if (!model || !content) return;
    updateModelScoringControls(model);
    const aggregation = document.getElementById('aggregation-select').value;
    const transitionScoreMode = document.getElementById('transition-score-select').value;
    const firstKey = document.getElementById('first-transition-select').value === 'true'
        ? 'included'
        : 'excluded';
    const scoringMetrics = model.synthetic_metrics?.scoring?.[transitionScoreMode]?.[firstKey]?.[aggregation]
        || (model.synthetic_metrics?.aggregations
            ? model.synthetic_metrics.aggregations[aggregation]
            : model.synthetic_metrics);
    const metrics = scoringMetrics;
    const trajectoryScoreDescription = model.architecture.trajectory_score
        || (model.model_type === 'lstm_forecaster'
            ? 'forecast error aggregation'
            : 'reconstruction aggregation');
    content.innerHTML = `
        <dl>
            <dt>Identifier</dt><dd>${model.model_id}</dd>
            <dt>Type</dt><dd>${model.model_type.replace(/_/g, ' ')}</dd>
            <dt>Feature set</dt><dd>${model.feature_set}</dd>
            <dt>Features</dt><dd>${model.features.join(', ')}</dd>
            <dt>Architecture</dt><dd>${Object.entries(model.architecture).map(([key, value]) => `${key}: ${value}`).join(', ')}</dd>
            <dt>Trajectory score</dt><dd>${formatPipelineValue(trajectoryScoreDescription)}</dd>
            <dt>Training data</dt><dd>${model.training.datasets.join(' + ')}</dd>
            <dt>Sampling</dt><dd>${formatPipelineValue(model.training.sampling_strategy || 'all')}</dd>
            <dt>Speed input</dt><dd>${formatPipelineValue(modelFeatureRepresentation(model))}</dd>
            <dt>Epochs</dt><dd>${model.training.epochs}</dd>
            <dt>Batch size</dt><dd>${model.training.batch_size}</dd>
            <dt>Final loss</dt><dd>${formatModelScore(model.final_loss)}</dd>
            <dt>Trusted-label AUC</dt><dd>${metrics ? Number(metrics.roc_auc).toFixed(3) : '--'}</dd>
        </dl>`;
}

function formatMetric(value) {
    const percentage = Number(value || 0) * 100;
    if (percentage > 0 && percentage < 0.1) return '<0.1%';
    return `${percentage.toFixed(1)}%`;
}

function balancedAccuracy(values) {
    if (Number.isFinite(Number(values.balanced_accuracy))) {
        return Number(values.balanced_accuracy);
    }
    return (Number(values.recall || 0) + 1 - Number(values.false_positive_rate || 0)) / 2;
}

function profileDisplayName(profile) {
    return profile.replace(/_/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase());
}

function updatePerformanceMetrics() {
    const model = allAvailableModels.find(item => item.model_id === currentModelSelection);
    const content = document.getElementById('performance-content');
    const cutoff = document.getElementById('performance-cutoff');
    const auc = document.getElementById('performance-auc');
    const percentage = parseFloat(document.getElementById('threshold-slider').value);
    const aggregation = document.getElementById('aggregation-select').value;
    const transitionScoreMode = document.getElementById('transition-score-select').value;
    const firstKey = document.getElementById('first-transition-select').value === 'true'
        ? 'included'
        : 'excluded';
    document.getElementById('metric-reference-label').textContent =
        'Threshold: held-out real scores excluding the reviewed negatives. Profile precision/F1 compare one profile with the full reviewed-negative cohort.';
    cutoff.textContent = `${percentage.toFixed(1)}% reference tail`;
    if (!model || !model.synthetic_metrics) {
        auc.textContent = 'AUC --';
        content.textContent = 'Synthetic metrics are not available for this model.';
        return;
    }
    if (model.synthetic_metrics.metrics_schema_version !== 3) {
        auc.textContent = 'AUC --';
        content.textContent = 'Synthetic metrics use an obsolete evaluation schema. Rebuild them.';
        return;
    }

    const scoringMetrics = model.synthetic_metrics?.scoring?.[transitionScoreMode]?.[firstKey]?.[aggregation]
        || (model.synthetic_metrics.aggregations
            ? model.synthetic_metrics.aggregations[aggregation]
            : model.synthetic_metrics);
    const metrics = scoringMetrics;
    if (!metrics || !metrics.threshold_metrics || !metrics.threshold_metrics.length) {
        auc.textContent = 'AUC --';
        content.textContent = 'Synthetic metrics are not available for this score aggregation.';
        return;
    }
    const selected = metrics.threshold_metrics.reduce((closest, candidate) =>
        Math.abs(candidate.anomaly_percentage - percentage)
            < Math.abs(closest.anomaly_percentage - percentage)
            ? candidate
            : closest
    );
    const selectedMetrics = selected;
    if (!selectedMetrics.overall || !selectedMetrics.profiles) {
        content.textContent = 'Rebuild synthetic metrics to display precision, recall, and F1.';
        return;
    }

    auc.textContent = `AUC ${Number(metrics.roc_auc).toFixed(3)}`;
    const rows = [
        ['All profiles', selectedMetrics.overall],
        ...Object.entries(selectedMetrics.profiles).map(([profile, values]) => [
            profileDisplayName(profile),
            values
        ])
    ];
    content.innerHTML = `
        <table class="metric-table">
            <thead><tr><th>Profile</th><th title="Depends on the synthetic-to-reviewed-negative sample ratio">Precision</th><th>Recall</th><th>F1</th><th>Balanced accuracy</th><th>FPR</th></tr></thead>
            <tbody>
                ${rows.map(([label, values]) => `
                    <tr>
                        <td>${label}</td>
                        <td>${formatMetric(values.precision)}</td>
                        <td>${formatMetric(values.recall)}</td>
                        <td>${formatMetric(values.f1_score)}</td>
                        <td>${formatMetric(balancedAccuracy(values))}</td>
                        <td>${formatMetric(values.false_positive_rate)}</td>
                    </tr>`).join('')}
            </tbody>
        </table>
        <div class="metric-counts">
            Labeled benchmark: ${selectedMetrics.overall.true_positives} TP ·
            ${selectedMetrics.overall.false_positives} FP ·
            ${selectedMetrics.overall.false_negatives} FN ·
            ${selectedMetrics.trusted_plausible?.true_negatives ?? '--'} TN
            ${scoringMetrics.paired_source_metrics
                ? ` · ${formatMetric(scoringMetrics.paired_source_metrics.overall.anomaly_above_source_rate)} score above its source`
                : ''}
        </div>`;
}



let plotlyShapesCache = {};
// Maps transition_id -> shape index, and observation_id -> shape index in the Plotly shapes array
let shapeIndexMap = {};

async function renderPlotlyProfile(user_id) {
    try {
        shapeIndexMap = {}; // Reset shape-to-index mapping for new user

        const userTransitions = appData.transitions_list.filter(t => String(t.user_id) === String(user_id));
        const userObscured = appData.obscured_observations.filter(o => String(o.user_id) === String(user_id));
        const userObservations = appData.observations.filter(o => String(o.user_id) === String(user_id));

        let transitionObsIds = new Set();
        for (let t of userTransitions) {
            transitionObsIds.add(String(t.observation_id1));
            transitionObsIds.add(String(t.observation_id2));
        }
        let isolatedObs = userObservations.filter(o => !transitionObsIds.has(String(o.observation_id)));

        let shapeList = [];
        let markerSize = 8.0;
        let dataEntryCounter = 0;
        let validDatesPlotly = new Set();

        // Group transitions by date
        let transitionsByDate = {};
        for (let t of userTransitions) {
            let d = t.date.trim();
            if (!transitionsByDate[d]) transitionsByDate[d] = [];
            transitionsByDate[d].push(t);
            validDatesPlotly.add(d.replace(/\//g, '-'));
        }

        // Group obscured by date
        let obscuredByDate = {};
        for (let o of userObscured) {
            let d = o.date.trim();
            if (!obscuredByDate[d]) obscuredByDate[d] = [];
            obscuredByDate[d].push(o);
            validDatesPlotly.add(d.replace(/\//g, '-'));
        }

        // Group isolated by date
        let isolatedByDate = {};
        for (let o of isolatedObs) {
            let d = o.date.trim().substring(0, 10).replace(/-/g, '/');
            if (!isolatedByDate[d]) isolatedByDate[d] = [];
            isolatedByDate[d].push(o);
            validDatesPlotly.add(d.replace(/\//g, '-'));
        }

        // Get all unique dates sorted
        let allDates = Array.from(new Set([...Object.keys(transitionsByDate), ...Object.keys(obscuredByDate), ...Object.keys(isolatedByDate)])).sort();

        for (let date of allDates) {
            dataEntryCounter += 1;
            let horizontalOffset = 0.0;

            let dateAnchor = date.replace(/\//g, '-');
            let trans = transitionsByDate[date] || [];
            let obs = obscuredByDate[date] || [];
            let isolated = isolatedByDate[date] || [];

            // Draw obscured shapes first (as squares)
            for (let o of obs) {
                let shapeIdx = shapeList.length;
                shapeIndexMap['obs_' + String(o.observation_id)] = shapeIdx;
                shapeList.push({
                    type: "rect",
                    xsizemode: 'pixel', ysizemode: 'pixel',
                    xanchor: dateAnchor,
                    yanchor: dataEntryCounter,
                    x0: markerSize * 0.5 + horizontalOffset, y0: -markerSize * 0.5,
                    x1: markerSize * 1.5 + horizontalOffset, y1: markerSize * 0.5,
                    line: { color: 'black', width: 1 },
                    fillcolor: 'white'
                });
                horizontalOffset += markerSize;
            }

            // Draw isolated shapes as green circles
            for (let o of isolated) {
                let shapeIdx = shapeList.length;
                shapeIndexMap['obs_' + String(o.observation_id)] = shapeIdx;
                shapeList.push({
                    type: "circle",
                    xsizemode: 'pixel', ysizemode: 'pixel',
                    xanchor: dateAnchor,
                    yanchor: dataEntryCounter,
                    x0: markerSize * 0.5 + horizontalOffset, y0: -markerSize * 0.5,
                    x1: markerSize * 1.5 + horizontalOffset, y1: markerSize * 0.5,
                    line: { width: 0 },
                    fillcolor: 'rgb(26, 150, 65)' // Green for isolated unobscured observations
                });
                horizontalOffset += markerSize;
            }

            // Draw first unobscured point if trans exists
            if (trans.length > 0) {
                let firstShapeIdx = shapeList.length;
                // Tag with the first observation_id of the first transition
                shapeIndexMap['obs_' + String(trans[0].observation_id1)] = firstShapeIdx;
                shapeList.push({
                    type: "circle",
                    xsizemode: 'pixel', ysizemode: 'pixel',
                    xanchor: dateAnchor,
                    yanchor: dataEntryCounter,
                    x0: markerSize * 0.5 + horizontalOffset, y0: -markerSize * 0.5,
                    x1: markerSize * 1.5 + horizontalOffset, y1: markerSize * 0.5,
                    line: { width: 0 },
                    fillcolor: 'rgb(128,128,128)' // First point is grey
                });
                horizontalOffset += markerSize;
            }

            // Draw remaining transitions
            for (let t of trans) {
                let additionalOffset = t.distance >= 0 ? Math.log2(t.distance + 1.0) : 0;
                horizontalOffset += additionalOffset;

                let fillcolorBySpeed = getSpeedColor(t.speed + " km/h");
                if (t.elapsed_time === 0) fillcolorBySpeed = 'rgb(0,0,0)'; // instantaneous

                let tShapeIdx = shapeList.length;
                // Tag with transition_id and also second observation_id
                shapeIndexMap['trans_' + String(t.transition_id)] = tShapeIdx;
                shapeIndexMap['obs_' + String(t.observation_id2)] = tShapeIdx;
                const isUnplausible = t.is_unplausible === true;
                shapeList.push({
                    type: "circle",
                    xsizemode: 'pixel', ysizemode: 'pixel',
                    xanchor: dateAnchor,
                    yanchor: dataEntryCounter,
                    x0: markerSize * 0.5 + horizontalOffset, y0: -markerSize * 0.5,
                    x1: markerSize * 1.5 + horizontalOffset, y1: markerSize * 0.5,
                    line: isUnplausible ? { color: '#ff4d4f', width: 2.5 } : { width: 0 },
                    fillcolor: fillcolorBySpeed,
                    isUnplausible: isUnplausible
                });
                horizontalOffset += markerSize;
            }
        }

        plotlyShapesCache[user_id] = shapeList;

        const dummyX = shapeList.map(s => s.xanchor);
        const dummyY = shapeList.map(s => s.yanchor);
        const hoverInfo = dummyX.map(x => validDatesPlotly.has(x) ? 'all' : 'skip');

        const dummyData = [{
            x: dummyX,
            y: dummyY,
            mode: 'markers',
            marker: { opacity: 0 },
            hoverinfo: hoverInfo
        }];

        let layout = {
            margin: { l: 80, r: 10, b: 40, t: 10, pad: 4 },
            font: { size: 10 },
            showlegend: false,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: {
                tickangle: 0,
                tickformat: '%Y',
                gridwidth: 0.5,
                gridcolor: 'rgb(230,230,230)',
            },
            yaxis: {
                range: [0, dataEntryCounter + 1.5],
                zeroline: true,
                zerolinewidth: 1,
                zerolinecolor: '#2a3f5f',
                gridwidth: 0.5,
                gridcolor: 'rgb(230,230,230)',
            },
            shapes: shapeList
        };

        Plotly.newPlot('mppPlotly', dummyData, layout, { responsive: true, scrollZoom: true });

        document.getElementById('mppPlotly').on('plotly_click', function (data) {
            if (data.points && data.points.length > 0) {
                const clickedDateString = data.points[0].x;
                if (!validDatesPlotly.has(clickedDateString)) return;

                const clickedDate = clickedDateString.replace(/-/g, '/');

                if (selectedDate.includes(clickedDate)) {
                    selectedDate = selectedDate.filter(d => d !== clickedDate);
                } else {
                    selectedDate.push(clickedDate);
                }
                console.log(`Plotly point clicked! Date: ${clickedDate}`);

                const allDays = document.querySelectorAll('.timeline-day');
                allDays.forEach(el => {
                    if (selectedDate.includes(el.textContent)) {
                        el.classList.add('selected');
                    } else {
                        el.classList.remove('selected');
                    }
                });

                updateMapForDate(user_id, selectedDate);
            }
        });

    } catch (error) {
        console.error('Error rendering Plotly Profile:', error);
    }
}

function highlightPlotlyDay(user_id, dates) {
    if (!plotlyShapesCache[user_id]) return;
    const shapes = plotlyShapesCache[user_id];
    const formattedDates = dates.map(d => d.replace(/\//g, '-'));

    // Create a new array of shapes to trigger relayout
    const updatedShapes = shapes.map(shape => {
        if (formattedDates.length === 0) {
            return shape; // Return to initial state
        } else if (formattedDates.includes(shape.xanchor)) {
            let border = { width: 0 };
            if (shape.fillcolor === 'white' || shape.fillcolor === 'rgb(255,255,255)') {
                border = { ...shape.line, color: 'black', width: 2 };
            } else if (shape.isUnplausible) {
                border = { color: '#ff4d4f', width: 2.5 };
            } else {
                border = { color: 'rgb(50,50,50)', width: 2 };
            }
            return {
                ...shape,
                opacity: 1,
                line: border
            };
        } else {
            return { ...shape, opacity: 0.15 };
        }
    });

    try {
        Plotly.relayout('mppPlotly', { shapes: updatedShapes });
    } catch (e) {
        console.error("Error highlighting Plotly day:", e);
    }
}

// Highlight a single shape on the Plotly profile when hovering a map element
let _lastHighlightedShapeIdx = null;
function highlightProfileShape(shapeIdx) {
    if (!currentUser || !plotlyShapesCache[currentUser]) return;
    if (_lastHighlightedShapeIdx === shapeIdx) return; // avoid redundant relayouts
    _lastHighlightedShapeIdx = shapeIdx;

    const shapes = plotlyShapesCache[currentUser];
    const formattedDates = selectedDate.map(d => d.replace(/\//g, '-'));

    const updatedShapes = shapes.map((shape, idx) => {
        let s = { ...shape };
        // Keep day-based dimming
        if (formattedDates.length > 0) {
            s.opacity = formattedDates.includes(shape.xanchor) ? 1 : 0.15;
        }
        // Add highlight ring on the hovered shape
        if (idx === shapeIdx) {
            s.line = { color: 'rgba(255, 0, 0, 1)', width: 3 };
        } else if (s.isUnplausible) {
            s.line = { color: '#ff4d4f', width: 2.5 };
        }
        return s;
    });

    try {
        Plotly.relayout('mppPlotly', { shapes: updatedShapes });
    } catch (e) { /* ignore */ }
}

function clearProfileHighlight() {
    if (_lastHighlightedShapeIdx === null) return;
    _lastHighlightedShapeIdx = null;
    // Re-apply the normal day highlighting
    if (currentUser) {
        highlightPlotlyDay(currentUser, selectedDate);
    }
}

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
            const polyline = L.polyline([[p1.lat, p1.lon], [p2.lat, p2.lon]], {
                color: isUnplausible ? '#ff4d4f' : color,
                weight: isUnplausible ? 6 : 4,
                opacity: 0.9,
                dashArray: isUnplausible ? '5, 5' : null
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
                            <polygon points="4,4 20,12 4,20" fill="${isUnplausible ? '#ff4d4f' : color}" stroke="white" stroke-width="2" />
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

async function setTimeline(user_id) {
    try {
        const activeDays = new Set();

        // Find all days with unobscured transitions for this user
        appData.transitions_list.forEach(t => {
            if (String(t.user_id) === String(user_id) && t.date.includes('/')) {
                activeDays.add(t.date.trim());
            }
        });

        // Add days with obscured observations
        appData.obscured_observations.forEach(o => {
            if (String(o.user_id) === String(user_id)) {
                let d = o.date.trim().substring(0, 10).replace(/-/g, '/');
                activeDays.add(d);
            }
        });

        // Add days with isolated observations
        appData.observations.forEach(o => {
            if (String(o.user_id) === String(user_id)) {
                let d = o.date.trim().substring(0, 10).replace(/-/g, '/');
                activeDays.add(d);
            }
        });

        globalDaysArray = Array.from(activeDays).sort();
        console.log(`User ${user_id} has observations on ${globalDaysArray.length} unique days.`);

        renderTimeline();

        return globalDaysArray;

    } catch (error) {
        console.error('Error generating timeline:', error);
        return [];
    }
}

// --- Timeline Framework ---

function buildTimelineTokens(mode, daysArray) {
    if (!daysArray || daysArray.length === 0) return [];

    if (mode === 'day') {
        return daysArray.map(day => ({ label: day, days: [day] }));
    }
    else if (mode === 'month') {
        const monthsMap = {};
        daysArray.forEach(day => {
            const parts = day.split('/');
            if (parts.length >= 2) {
                const monthLabel = parts[0] + '/' + parts[1];
                if (!monthsMap[monthLabel]) monthsMap[monthLabel] = [];
                monthsMap[monthLabel].push(day);
            }
        });
        return Object.keys(monthsMap).sort().map(month => ({ label: month, days: monthsMap[month] }));
    }
    else if (mode === 'year') {
        const yearsMap = {};
        daysArray.forEach(day => {
            const parts = day.split('/');
            if (parts.length >= 1) {
                const yearLabel = parts[0];
                if (!yearsMap[yearLabel]) yearsMap[yearLabel] = [];
                yearsMap[yearLabel].push(day);
            }
        });
        return Object.keys(yearsMap).sort().map(year => ({ label: year, days: yearsMap[year] }));
    }
    else if (mode === 'smart') {
        // This mode clusters observations that were done in maximum 5 days in a row, and returns the 5 most important clusters
        let clusters = [];
        let currentCluster = { label: '', days: [] };
        let prevDate = null;

        daysArray.forEach(day => {
            const currDate = new Date(day.replace(/\//g, '-'));
            if (!prevDate) {
                currentCluster.days.push(day);
            } else {
                const diffTime = Math.abs(currDate - prevDate);
                const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

                if (diffDays > 5) {
                    clusters.push(currentCluster);
                    currentCluster = { label: '', days: [day] };
                } else {
                    currentCluster.days.push(day);
                }
            }
            prevDate = currDate;
        });
        if (currentCluster.days.length > 0) {
            clusters.push(currentCluster);
        }

        clusters.sort((a, b) => b.days.length - a.days.length);
        const topClusters = clusters.slice(0, 5);
        topClusters.sort((a, b) => a.days[0].localeCompare(b.days[0]));

        return topClusters.map(cluster => {
            const startNode = cluster.days[0];
            const endNode = cluster.days[cluster.days.length - 1];
            return {
                label: startNode === endNode ? startNode : `${startNode} to ${endNode}`,
                days: cluster.days
            };
        });
    }
    return [];
}

function renderTimeline() {
    const timelineList = document.getElementById('timeline');
    const modeSelect = document.getElementById('timeline-mode');
    if (!timelineList || !modeSelect) return;

    timelineList.innerHTML = '';
    const mode = modeSelect.value;
    const tokens = buildTimelineTokens(mode, globalDaysArray);

    tokens.forEach(token => {
        const li = document.createElement('li');
        li.className = 'timeline-day';
        li.textContent = token.label;

        const tokenTrajectories = appData.trajectories_list.filter(traj =>
            String(traj.user_id) === String(currentUser)
            && token.days.includes(traj.date.trim())
        );
        const hasBaselineAnomaly = tokenTrajectories.some(traj => traj.baseline_unplausible);
        const hasModelAnomaly = tokenTrajectories.some(traj => traj.model_unplausible);
        const baselineCount = tokenTrajectories.filter(
            traj => traj.baseline_unplausible
        ).length;
        const modelCount = tokenTrajectories.filter(
            traj => traj.model_unplausible
        ).length;
        li.title = `${baselineCount} physical-rule violation(s), ${modelCount} model detection(s)`;
        if (hasBaselineAnomaly && hasModelAnomaly) {
            li.classList.add('combined-anomaly');
        } else if (hasBaselineAnomaly) {
            li.classList.add('baseline-anomaly');
        } else if (hasModelAnomaly) {
            li.classList.add('model-anomaly');
        }

        const isFullySelected = token.days.length > 0 && token.days.every(d => selectedDate.includes(d));
        if (isFullySelected) {
            li.classList.add('selected');
        }

        li.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDraggingTimeline = true;

            const currentlySelected = token.days.every(d => selectedDate.includes(d));
            if (currentlySelected) {
                selectedDate = selectedDate.filter(d => !token.days.includes(d));
                li.classList.remove('selected');
            } else {
                token.days.forEach(d => {
                    if (!selectedDate.includes(d)) selectedDate.push(d);
                });
                li.classList.add('selected');
            }
        });

        li.addEventListener('mouseenter', () => {
            if (isDraggingTimeline) {
                const currentlySelected = token.days.every(d => selectedDate.includes(d));
                if (currentlySelected) {
                    selectedDate = selectedDate.filter(d => !token.days.includes(d));
                    li.classList.remove('selected');
                } else {
                    token.days.forEach(d => {
                        if (!selectedDate.includes(d)) selectedDate.push(d);
                    });
                    li.classList.add('selected');
                }
            }
        });

        timelineList.appendChild(li);
    });
}

async function loadObservationsList() {
    try {
        const response = await fetch('../observations_list.txt');
        if (!response.ok) throw new Error('observations_list.txt not found');
        const text = await response.text();
        const files = text.split('\n').filter(line => line.trim() !== '');
        console.log(`Loaded ${files.length} observation files:`, files);
        return files;
    } catch (error) {
        console.error('Error loading observations list:', error);
        return [];
    }
}

// Automatically load and populate the user-select dropdown when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    // 0. Initialize the Map
    leafletMap = L.map('map').setView([0, 0], 2); // Default to a global view

    // Log zoom level and hide arrows when zoomed out to improve performance
    function checkZoomLevel() {
        const currentZoom = leafletMap.getZoom();
        console.log(`Current map zoom level: ${currentZoom}`);
        if (currentZoom < 6) {
            leafletMap.getContainer().classList.add('hide-arrows');
        } else {
            leafletMap.getContainer().classList.remove('hide-arrows');
        }
    }
    leafletMap.on('zoomend', checkZoomLevel);
    checkZoomLevel(); // Initial check

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(leafletMap);

    // 0.5 Build Legend dynamically
    let legendControl = null;
    function updateLegend() {
        if (legendControl) leafletMap.removeControl(legendControl);

        legendControl = L.control({ position: 'bottomright' });
        legendControl.onAdd = function (map) {
            const div = L.DomUtil.create('div', 'info legend');
            const p = isColorblindMode ? colorPaletteColorblind : colorPaletteStandard;
            const categories = [
                { threshold: '0 km/h', color: p.c0 },
                { threshold: '< 5 km/h', color: p.c5 },
                { threshold: '< 10 km/h', color: p.c10 },
                { threshold: '< 25 km/h', color: p.c25 },
                { threshold: '< 80 km/h', color: p.c80 },
                { threshold: '< 200 km/h', color: p.c200 },
                { threshold: '>= 200 km/h', color: p.cMax }
            ];
            div.innerHTML = '<strong>Speed Key</strong><br>';
            for (let i = 0; i < categories.length; i++) {
                div.innerHTML +=
                    '<i style="background-color:' + categories[i].color + '; display: inline-block; width: 50px; height: 16px; float: left; margin-right: 8px; margin-top: 3px; border-radius: 3px; border: 1px solid rgba(0, 0, 0, 0.2);"></i> ' +
                    categories[i].threshold + '<br>';
            }
            return div;
        };
        legendControl.addTo(leafletMap);
    }

    updateLegend();

    // Colorblind mode toggle listener
    const colorblindToggle = document.getElementById('colorblind-toggle');
    if (colorblindToggle) {
        colorblindToggle.addEventListener('change', (e) => {
            isColorblindMode = e.target.checked;
            updateLegend();

            // Clear Plotly cache to force recalculation of shape colors
            plotlyShapesCache = {};

            // Re-render visuals if a user is already active
            if (currentUser) {
                renderPlotlyProfile(currentUser);
                if (selectedDate) {
                    updateMapForDate(currentUser, selectedDate);
                }
            }
        });
    }

    // Timeline resolution toggle listener
    const timelineModeSelect = document.getElementById('timeline-mode');
    if (timelineModeSelect) {
        timelineModeSelect.addEventListener('change', () => {
            renderTimeline();
        });
    }

    let usernames = {};
    try {
        const statsResponse = await fetch('ressources/usernames.json');
        if (statsResponse.ok) {
            usernames = await statsResponse.json();
        }
    } catch (e) { console.warn("Could not load usernames"); }

    const selectElement = document.getElementById('user-select');
    if (!selectElement) {
        console.warn('Could not find the element with ID "user-select"');
        return;
    }

    // Add a 'change' event listener to the <select> element itself
    selectElement.addEventListener('change', (event) => {
        const selectedUserId = event.target.value;
        if (selectedUserId) {
            setUser(selectedUserId); 
        }
    });

    async function initializeDataset() {
        // Clear UI state
        currentUser = null;
        selectedDate = [];
        originalTransitionsData = {};
        originalTrajectoriesData = {};
        if (leafletMap) {
            currentMarkers.forEach(m => leafletMap.removeLayer(m));
            currentMarkers = [];
            if (window.routePolyline) leafletMap.removeLayer(window.routePolyline);
        }
        document.getElementById('timeline').innerHTML = '';
        Plotly.purge('mppPlotly');
        
        selectElement.innerHTML = '<option value="">--Please choose a user--</option>';

        const users = await fetchAndParseUsers();
        
        let userCounts = {};
        if (appData && appData.users_list) {
            for (const u of appData.users_list) {
                userCounts[String(u.username)] = u.nb_observations || 0;
            }
        }

        users.sort((a, b) => {
            const countA = userCounts[a] || 0;
            const countB = userCounts[b] || 0;
            return countB - countA;
        });

        const topUsers = users.slice(0, 20000);
        const fragment = document.createDocumentFragment();

        for (const user of topUsers) {
            const option = document.createElement('option');
            option.value = user;
            // For Gowalla, there are no usernames, just user IDs. We check currentDatabase.
            const uname = (currentDatabase === 'original' && usernames[user]) ? usernames[user] :
                          (currentDatabase === 'synthetic' ? user.replace(/_/g, ' ') : `User ${user}`);
            const count = userCounts[user] || 0;
            option.textContent = `${uname} (${count} obs)`;
            fragment.appendChild(option);
        }
        selectElement.appendChild(fragment);
    }

    const databaseSelect = document.getElementById('database-select');
    if (databaseSelect) {
        databaseSelect.addEventListener('change', (e) => {
            currentDatabase = e.target.value;
            updateModelDropdown();
            initializeDataset();
        });
    }

    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', async (e) => {
            currentModelSelection = e.target.value;
            await loadModelMetrics(currentModelSelection);
            updateModelInformation();
            updatePerformanceMetrics();
            if (currentUser) {
                await fetchModelScoresForUser(currentUser);
                setTimeline(currentUser);
                renderPlotlyProfile(currentUser);
                if (selectedDate.length > 0) {
                    updateMapForDate(currentUser, selectedDate);
                }
            }
        });
    }

    const thresholdSlider = document.getElementById('threshold-slider');
    if (thresholdSlider) {
        let thresholdTimer = null;
        thresholdSlider.addEventListener('input', async (e) => {
            document.getElementById('threshold-percentage').textContent = `${parseFloat(e.target.value).toFixed(1)}%`;
            updatePerformanceMetrics();
            clearTimeout(thresholdTimer);
            thresholdTimer = setTimeout(async () => {
            if (currentUser) {
                await fetchModelScoresForUser(currentUser);
                setTimeline(currentUser);
                renderPlotlyProfile(currentUser);
                if (selectedDate.length > 0) {
                    updateMapForDate(currentUser, selectedDate);
                }
            }
            }, 100);
        });
    }

    [
        'aggregation-select',
        'transition-score-select',
        'first-transition-select',
    ].forEach(id => {
        document.getElementById(id).addEventListener('change', async () => {
            updateModelInformation();
            updatePerformanceMetrics();
            if (currentUser) {
                await fetchModelScoresForUser(currentUser);
                setTimeline(currentUser);
                renderPlotlyProfile(currentUser);
                if (selectedDate.length > 0) {
                    updateMapForDate(currentUser, selectedDate);
                }
            }
        });
    });

    const modelSetupButton = document.getElementById('model-setup-button');
    const modelSetupPanel = document.getElementById('model-setup-panel');
    const closeModelSetup = document.getElementById('close-model-setup');
    const statsButton = document.getElementById('stats-button');
    const statsPanel = document.getElementById('stats-panel');
    const closeStats = document.getElementById('close-stats');
    modelSetupButton.addEventListener('click', () => {
        modelSetupPanel.hidden = !modelSetupPanel.hidden;
    });
    closeModelSetup.addEventListener('click', () => {
        modelSetupPanel.hidden = true;
    });
    statsButton.addEventListener('click', async () => {
        statsPanel.hidden = !statsPanel.hidden;
        if (!statsPanel.hidden) {
            await loadDatasetStats();
        }
    });
    closeStats.addEventListener('click', () => {
        statsPanel.hidden = true;
    });
    ['stats-dataset', 'stats-subset', 'stats-metric'].forEach(id => {
        document.getElementById(id).addEventListener('change', async () => {
            try {
                await loadDatasetStats();
            } catch (error) {
                document.getElementById('stats-summary').textContent = error.message;
            }
        });
    });
    document.getElementById('export-stats-json').addEventListener('click', () => {
        if (!currentStatsPayload) return;
        const name = `${currentStatsPayload.dataset}_${currentStatsPayload.subset}_${currentStatsPayload.metric}`;
        downloadStatsFile(`${name}.json`, JSON.stringify(currentStatsPayload, null, 2), 'application/json');
    });
    document.getElementById('export-stats-csv').addEventListener('click', () => {
        if (!currentStatsPayload) return;
        const rows = ['bin_left,bin_right,count,comparison_count'].concat(
            currentStatsPayload.histogram.map((row, index) =>
                `${row.bin_left},${row.bin_right},${row.count},${currentStatsPayload.comparison_histogram?.[index]?.count ?? ''}`
            )
        );
        const name = `${currentStatsPayload.dataset}_${currentStatsPayload.subset}_${currentStatsPayload.metric}`;
        downloadStatsFile(`${name}.csv`, rows.join('\n'), 'text/csv');
    });
    document.getElementById('export-stats-png').addEventListener('click', () => {
        if (!currentStatsPayload) return;
        Plotly.downloadImage('stats-plot', {
            format: 'png',
            filename: `${currentStatsPayload.dataset}_${currentStatsPayload.subset}_${currentStatsPayload.metric}`,
            width: 1200,
            height: 700
        });
    });
    [
        'filter-model-type',
        'filter-training-mode',
        'filter-feature-representation'
    ].forEach(id => {
        document.getElementById(id).addEventListener('change', async () => {
            updateModelDropdown();
            if (currentUser && currentModelSelection) {
                await fetchModelScoresForUser(currentUser);
                setTimeline(currentUser);
                renderPlotlyProfile(currentUser);
                if (selectedDate.length > 0) {
                    updateMapForDate(currentUser, selectedDate);
                }
            }
        });
    });

    // Initial load. Query parameters are also useful for reproducible visual checks.
    loadAvailableModels().then(async () => {
        populateModelFilters();
        updateModelDropdown();
        const params = new URLSearchParams(window.location.search);
        const requestedDataset = params.get('dataset');
        const requestedUser = params.get('user');
        if (requestedDataset) {
            currentDatabase = requestedDataset === 'inat' ? 'original' : requestedDataset;
            databaseSelect.value = currentDatabase;
            updateModelDropdown();
            if (currentDatabase === 'synthetic') {
                modelSetupPanel.hidden = false;
            }
        }
        await initializeDataset();
        if (requestedUser) {
            selectElement.value = requestedUser;
            if (selectElement.value) await setUser(requestedUser);
        }
    });
});

function toggleProfile() {
    const chart = document.getElementById('chart-container');
    const btn = document.getElementById('toggle-profile');
    if (chart.classList.contains('hidden')) {
        chart.classList.remove('hidden');
        chart.style.width = ''; // Clear inline width so it snaps back to CSS default or previously expanded width
        btn.textContent = '▶';
        setTimeout(() => {
            if (window.Plotly) Plotly.Plots.resize('mppPlotly');
            if (leafletMap) leafletMap.invalidateSize();
        }, 300);
    } else {
        chart.classList.add('hidden');
        btn.textContent = '◀';
        setTimeout(() => {
            if (leafletMap) leafletMap.invalidateSize();
        }, 300);
    }
}

// Resizer logic
document.addEventListener('DOMContentLoaded', () => {
    const resizer = document.getElementById('resizer');
    const chart = document.getElementById('chart-container');
    const container = document.getElementById('content-split');

    if (!resizer || !chart || !container) return;

    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        document.body.style.cursor = 'col-resize';
        chart.style.transition = 'none'; // Prevent CSS transition lagging during manual resize
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const containerRect = container.getBoundingClientRect();
        let newWidth = containerRect.right - e.clientX;

        if (newWidth < 50) {
            newWidth = 0;
            if (!chart.classList.contains('hidden')) {
                chart.classList.add('hidden');
                document.getElementById('toggle-profile').textContent = '◀';
            }
        } else {
            if (chart.classList.contains('hidden')) {
                chart.classList.remove('hidden');
                chart.style.width = ''; // Clear inline width to let it recover visually
                document.getElementById('toggle-profile').textContent = '▶';
            }
            if (newWidth > containerRect.width - 200) newWidth = containerRect.width - 200;
        }

        chart.style.width = newWidth + 'px';
        chart.style.flex = 'none'; // Ensure CSS flex-basis doesn't fight our inline width

        if (window.Plotly) Plotly.Plots.resize('mppPlotly');
        if (leafletMap) leafletMap.invalidateSize();
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            chart.style.transition = ''; // Restore transition
            if (leafletMap) leafletMap.invalidateSize();
        }
    });
});
