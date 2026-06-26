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

function currentScoringSettings() {
    const includeFirstTransition = document.getElementById('first-transition-select').value === 'true';
    return {
        aggregation: document.getElementById('aggregation-select').value,
        transition_score_mode: document.getElementById('transition-score-select').value,
        include_first_transition: includeFirstTransition,
        transition_score_features: selectedTransitionScoreFeatures()
    };
}

function scoringMetricsKey(settings) {
    return JSON.stringify({
        aggregation: settings.aggregation,
        transition_score_mode: settings.transition_score_mode,
        include_first_transition: settings.include_first_transition,
        transition_score_features: settings.transition_score_features || []
    });
}

function selectedSyntheticMetrics(model) {
    if (!model) return null;
    const settings = currentScoringSettings();
    if (settings.transition_score_features) {
        return model.synthetic_metric_variants?.[scoringMetricsKey(settings)] || null;
    }
    const firstKey = settings.include_first_transition ? 'included' : 'excluded';
    return model.synthetic_metrics?.scoring?.[settings.transition_score_mode]?.[firstKey]?.[settings.aggregation]
        || (model.synthetic_metrics?.aggregations
            ? model.synthetic_metrics.aggregations[settings.aggregation]
            : model.synthetic_metrics);
}

async function loadCurrentScoringMetrics() {
    const model = allAvailableModels.find(item => item.model_id === currentModelSelection);
    if (!model) return null;
    await loadModelMetrics(model.model_id);
    const settings = currentScoringSettings();
    if (!settings.transition_score_features) return selectedSyntheticMetrics(model);
    const key = scoringMetricsKey(settings);
    model.synthetic_metric_variants ||= {};
    if (model.synthetic_metric_variants[key]) return model.synthetic_metric_variants[key];
    model.synthetic_metric_variant_promises ||= {};
    if (model.synthetic_metric_variant_promises[key]) {
        return model.synthetic_metric_variant_promises[key];
    }
    const params = new URLSearchParams({
        model_id: model.model_id,
        aggregation: settings.aggregation,
        transition_score_mode: settings.transition_score_mode,
        include_first_transition: String(settings.include_first_transition),
        transition_score_features: settings.transition_score_features.join(',')
    });
    model.synthetic_metric_variant_promises[key] = fetch(`/api/model_metrics?${params.toString()}`)
        .then(async response => {
            if (!response.ok) return null;
            model.synthetic_metric_variants[key] = await response.json();
            return model.synthetic_metric_variants[key];
        })
        .finally(() => {
            delete model.synthetic_metric_variant_promises[key];
        });
    return model.synthetic_metric_variant_promises[key];
}

function refreshModelPanels() {
    updateModelInformation();
    updatePerformanceMetrics();
    loadCurrentScoringMetrics().then(metrics => {
        if (!metrics) return;
        updateModelInformation();
        updatePerformanceMetrics();
    });
}

function updateModelDropdown() {
    const modelSelect = document.getElementById('model-select');
    if (!modelSelect) return;
    
    const dbPrefix = currentDatasetName();
    
    const previousSelection = currentModelSelection;
    modelSelect.innerHTML = '<option value="">No model selected</option>';
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
    if (previousSelection && relevantModels.some(model => model.model_id === previousSelection)) {
        modelSelect.value = previousSelection;
    }
    currentModelSelection = modelSelect.value || '';
    refreshModelPanels();
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

function usesNativeTrajectoryScore(model) {
    return Boolean(model?.architecture?.trajectory_score);
}

function selectedTransitionScoreFeatures() {
    const model = allAvailableModels.find(item => item.model_id === currentModelSelection);
    const options = [...document.querySelectorAll('#transition-feature-options input[type="checkbox"]')];
    if (!model || !options.length || usesNativeTrajectoryScore(model)) return null;
    const selected = options.filter(option => option.checked).map(option => option.value);
    if (!selected.length || selected.length === model.features.length) return null;
    return selected;
}

function populateTransitionFeatureOptions(model) {
    const container = document.getElementById('transition-feature-options');
    if (!container || !model || transitionFeatureModelId === model.model_id) return;
    transitionFeatureModelId = model.model_id;
    container.innerHTML = model.features.map(feature => `
        <label class="feature-toggle">
            <input type="checkbox" value="${feature}" checked>
            <span>${formatPipelineValue(feature)}</span>
        </label>
    `).join('');
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
    const featureControl = document.getElementById('transition-feature-control');
    if (featureControl) featureControl.classList.toggle('disabled', nativeScore);
    document
        .querySelectorAll('#transition-feature-options input[type="checkbox"]')
        .forEach(option => {
            option.disabled = nativeScore;
            if (nativeScore) option.checked = true;
        });
}

function updateModelInformation() {
    const model = allAvailableModels.find(item => item.model_id === currentModelSelection);
    const content = document.getElementById('model-info-content');
    if (!content) return;
    if (!model) {
        content.textContent = 'No model selected.';
        updateModelScoringControls(null);
        return;
    }
    populateTransitionFeatureOptions(model);
    updateModelScoringControls(model);
    const scoringMetrics = selectedSyntheticMetrics(model);
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
    const scoringMetrics = selectedSyntheticMetrics(model);
    const metrics = scoringMetrics;
    if (!metrics || !metrics.threshold_metrics || !metrics.threshold_metrics.length) {
        auc.textContent = 'AUC --';
        content.textContent = 'Loading synthetic metrics for this score setting.';
        loadCurrentScoringMetrics().then(() => {
            updateModelInformation();
            updatePerformanceMetrics();
        });
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
