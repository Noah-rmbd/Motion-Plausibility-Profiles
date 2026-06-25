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
