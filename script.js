/**
 * Parses the content of frequent-poster.log and returns an array of user IDs.
 * Detects user_id by looking for a number followed by ", active from".
 * 
 * @param {string} logContent - The raw text content of the log file.
 * @returns {string[]} An array of user IDs found in the log.
 */
function parseUsersFromLog(logContent) {
    const userIds = [];
    // Split the content by newline to process line by line
    const lines = logContent.split('\n');

    for (const line of lines) {
        // Find lines that contain the target string
        const targetString = ', active from';
        if (line.includes(targetString)) {
            // Extract the user ID part before the target string
            // E.g., from "5129707, active from..." we get "5129707"
            const userId = line.split(targetString)[0].trim();

            // Verify if the extracted part is a number to be safe
            if (!isNaN(userId) && userId !== '') {
                userIds.push(userId);
            }
        }
    }

    return userIds;
}

/**
 * Fetches the frequent-poster.log file, parses it, and returns the list of users.
 * To be used in a browser environment.
 * 
 * @returns {Promise<string[]>} A promise that resolves to an array of user IDs.
 */
async function fetchAndParseUsers() {
    try {
        const response = await fetch('frequent-poster.log');
        if (!response.ok) {
            throw new Error(`Failed to fetch log file: ${response.status} ${response.statusText}`);
        }

        const logContent = await response.text();
        const users = parseUsersFromLog(logContent);

        console.log(`Found ${users.length} users in the log file.`);
        return users;
    } catch (error) {
        console.error('Error fetching or parsing frequent-poster.log:', error);
        return [];
    }
}

// Store the currently selected user
let currentUser = null;
let selectedDate = null;
let leafletMap = null;
let currentMarkers = [];

// Set the map, the timeline and load the pdf files
function setUser(user_id) {
    currentUser = user_id;
    console.log(`Great news ${user_id}`);
    findImages(user_id);
    setTimeline(user_id);
}

function findImages(user_id) {

}

// Function to get color from speed
function getSpeedColor(speedStr) {
    if (!speedStr) return 'gray';
    const numMatch = speedStr.match(/[\d.]+/);
    if (!numMatch) return 'gray';
    const speed = parseFloat(numMatch[0]);

    if (speed === 0) return 'rgb(0, 0, 0)'; // Black
    if (speed < 5) return 'rgb(26, 150, 65)'; // Green
    if (speed < 10) return 'rgb(166, 217, 106)'; // Light green
    if (speed < 25) return 'rgb(203, 203, 15)'; // Yellow
    if (speed < 80) return 'rgb(253, 174, 97)'; // Orange
    if (speed < 200) return 'rgb(215, 25, 28)'; // Red
    return 'rgb(129, 15, 124)'; // Purple
}

// Function for async sleep
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Read log again to find observations for that user on that specific date
async function updateMapForDate(user_id, date) {
    try {
        const response = await fetch('frequent-poster.log');
        const text = await response.text();
        const lines = text.split('\n');

        // Clear previously displayed markers
        currentMarkers.forEach(marker => leafletMap.removeLayer(marker));
        currentMarkers = [];

        let inUserBlock = false;
        let bounds = L.latLngBounds();
        let sequence = [];

        for (const line of lines) {
            // Skip separators
            if (line.startsWith('========')) continue;

            // Reached user header
            if (line.includes(', active from')) {
                const currentId = line.split(',')[0].trim();
                inUserBlock = (currentId === String(user_id));
                continue;
            }

            // If in target user block, look for matching points
            if (inUserBlock && line.trim() !== '') {
                const parts = line.split(',');
                if (parts.length >= 4) {
                    const id = parts[0].trim();
                    const lat = parseFloat(parts[1].trim());
                    const lon = parseFloat(parts[2].trim());
                    const obsDate = parts[3].trim();

                    // Skip if ID starts with 'iN-o', or if the date doesn't match
                    if (obsDate === date && !id.startsWith('iN-o')) {
                        sequence.push({ lat, lon, id, parts });
                    }
                }
            }
        }

        if (sequence.length > 0) {
            // First drop all markers and calculate bounds
            for (let pt of sequence) {
                const marker = L.marker([pt.lat, pt.lon]).addTo(leafletMap);
                const time = pt.parts.length > 4 ? pt.parts[4].trim() : '';
                marker.bindPopup(`<b>ID:</b> ${pt.id}<br><b>Time:</b> ${time}`);
                currentMarkers.push(marker);
                bounds.extend([pt.lat, pt.lon]);
            }

            // Set map view to show all points optimally before drawing connecting lines
            leafletMap.fitBounds(bounds, { padding: [50, 50] });

            // Check if animation is toggled
            const isAnimated = document.getElementById('animate-path') && document.getElementById('animate-path').checked;

            // Wait briefly to let the camera settle to the new bounds
            if (isAnimated) await sleep(800);

            // Draw paths progressively
            for (let i = 1; i < sequence.length; i++) {
                let prev = sequence[i - 1];
                let curr = sequence[i];

                let speedStr = curr.parts.find(p => p.includes('km/h'));
                let color = getSpeedColor(speedStr);

                const polyline = L.polyline([[prev.lat, prev.lon], [curr.lat, curr.lon]], {
                    color: color,
                    weight: 4,
                    opacity: 0.8
                }).addTo(leafletMap);

                currentMarkers.push(polyline);

                // Add a native directional arrow at the midpoint of the segment
                const p1 = leafletMap.project([prev.lat, prev.lon]);
                const p2 = leafletMap.project([curr.lat, curr.lon]);
                const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180 / Math.PI;
                const midLat = (prev.lat + curr.lat) / 2;
                const midLon = (prev.lon + curr.lon) / 2;

                const arrowIcon = L.divIcon({
                    className: 'custom-arrow-icon',
                    html: `
                        <div style="width: 20px; height: 20px; transform: rotate(${angle}deg);">
                            <svg viewBox="0 0 24 24" width="20" height="20" style="overflow: visible;">
                                <polygon points="4,4 20,12 4,20" fill="${color}" stroke="white" stroke-width="2" />
                            </svg>
                        </div>
                    `,
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                });

                const arrowMarker = L.marker([midLat, midLon], { icon: arrowIcon, interactive: false }).addTo(leafletMap);
                currentMarkers.push(arrowMarker);

                if (isAnimated) {
                    await sleep(400); // progressive drawing delay
                }
            }
        } else {
            console.log("No valid points to display for this date.");
        }
    } catch (error) {
        console.error('Error updating map:', error);
    }
}

async function setTimeline(user_id) {
    try {
        const response = await fetch('frequent-poster.log');
        if (!response.ok) {
            throw new Error(`Failed to fetch log file: ${response.status} ${response.statusText}`);
        }

        const text = await response.text();
        const lines = text.split('\n');

        let inUserBlock = false;
        const activeDays = new Set();

        for (const line of lines) {
            // Skip the separator lines
            if (line.startsWith('========')) continue;

            // Check if this is a header for a user block
            if (line.includes(', active from')) {
                const currentId = line.split(',')[0].trim();
                inUserBlock = (currentId === String(user_id));
                continue; // Move to the next line (which will be a separator)
            }

            // If we are currently inside the target user's block, parse the observation
            if (inUserBlock && line.trim() !== '') {
                // Example format: iN-p110837819, -32.11, 116.15, 2014/10/18, 23:48:00, ...
                const parts = line.split(',');
                if (parts.length >= 4) {
                    const dateField = parts[3].trim();
                    // Basic check to ensure it's a date field containing a slash
                    if (dateField.includes('/')) {
                        activeDays.add(dateField);
                    }
                }
            }
        }

        const daysArray = Array.from(activeDays);
        console.log(`User ${user_id} has observations on ${daysArray.length} unique days.`);

        // Populate the timeline at the bottom of the page
        const timelineList = document.getElementById('timeline');
        if (timelineList) {
            // Clear out any previous days first
            timelineList.innerHTML = '';

            daysArray.forEach(dateStr => {
                const li = document.createElement('li');
                li.className = 'timeline-day';
                li.textContent = dateStr;

                // Make each day clickable
                li.addEventListener('click', () => {
                    // Remove 'selected' class from all days
                    const allDays = document.querySelectorAll('.timeline-day');
                    allDays.forEach(el => el.classList.remove('selected'));

                    // Add 'selected' class to the clicked day
                    li.classList.add('selected');

                    // Store it globally (we will declare this at the top of the file)
                    selectedDate = dateStr;
                    console.log('User clicked date:', selectedDate);

                    // Update map markers
                    updateMapForDate(user_id, selectedDate);
                });

                timelineList.appendChild(li);
            });
        }

        return daysArray;

    } catch (error) {
        console.error('Error fetching or parsing frequent-poster.log:', error);
        return [];
    }
}

// Automatically load and populate the user-select dropdown when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    // 0. Initialize the Map
    leafletMap = L.map('map').setView([0, 0], 2); // Default to a global view
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(leafletMap);

    // 0.5 Build Legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function (map) {
        const div = L.DomUtil.create('div', 'info legend');
        const categories = [
            { threshold: '0 km/h', color: 'rgb(0, 0, 0)' },
            { threshold: '< 5 km/h', color: 'rgb(26, 150, 65)' },
            { threshold: '< 10 km/h', color: 'rgb(166, 217, 106)' },
            { threshold: '< 25 km/h', color: 'rgb(203, 203, 15)' },
            { threshold: '< 80 km/h', color: 'rgb(253, 174, 97)' },
            { threshold: '< 200 km/h', color: 'rgb(215, 25, 28)' },
            { threshold: '>= 200 km/h', color: 'rgb(129, 15, 124)' }
        ];
        div.innerHTML = '<strong>Speed Key</strong><br>';
        for (let i = 0; i < categories.length; i++) {
            div.innerHTML +=
                '<i style="background-color:' + categories[i].color + '; display: inline-block; width: 50px; height: 16px; float: left; margin-right: 8px; margin-top: 3px; border-radius: 3px; border: 1px solid rgba(0, 0, 0, 0.2);"></i> ' +
                categories[i].threshold + '<br>';
        }
        return div;
    };
    legend.addTo(leafletMap);

    // 1. Fetch the user list
    const users = await fetchAndParseUsers();

    // 2. Identify the select element
    const selectElement = document.getElementById('user-select');
    if (!selectElement) {
        console.warn('Could not find the element with ID "user-select"');
        return;
    }

    // Add a 'change' event listener to the <select> element itself
    selectElement.addEventListener('change', (event) => {
        // event.target.value contains the value of the selected option
        const selectedUserId = event.target.value;
        if (selectedUserId) {
            setUser(selectedUserId); // Pass the selected user to your function
        }
    });

    // 3. Append each user as a new <option>
    for (const user of users) {
        const option = document.createElement('option');
        option.value = user;
        option.textContent = `User ${user}`;
        selectElement.appendChild(option);
    }
});
