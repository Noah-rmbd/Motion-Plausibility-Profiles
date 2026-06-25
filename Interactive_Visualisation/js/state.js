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
let currentStatsPayload = null;
const visibleDatasets = new Set(['inat', 'gowalla', 'synthetic']);
let transitionFeatureModelId = '';
let latestScoreRequestKey = '';

function currentDatasetName() {
    return currentDatabase === 'original' ? 'inat' : currentDatabase;
}

function modelUsesVisibleDatasetsOnly(model) {
    const trainingDatasets = model.training?.datasets || [];
    const evaluatedDatasets = model.datasets?.map(dataset => dataset.dataset) || [];
    return trainingDatasets.every(dataset => visibleDatasets.has(dataset))
        && evaluatedDatasets.every(dataset => visibleDatasets.has(dataset));
}


// Active user and map state shared across the visualization modules.
let currentUser = null;
let selectedDate = [];
let leafletMap = null;
let currentMarkers = [];
let isDraggingTimeline = false;
let globalDaysArray = [];
