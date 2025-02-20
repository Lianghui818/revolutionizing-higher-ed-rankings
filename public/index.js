let data = [];
let categories = [];
let currentPage = 1;
const rowsPerPage = 25; // Set the number of rows per page
let selectedRegion = 'all'; // Default to 'all' regions

async function loadCSV(filePath) {
    const response = await fetch(filePath);
    const csvText = await response.text();
    
    const rows = csvText.trim().split('\n');
    const headers = rows.shift().split(',');

    return rows.map(row => {
        const rowData = row.split(',');
        return headers.reduce((obj, header, index) => {
            obj[header.trim()] = rowData[index]?.trim() || '';
            return obj;
        }, {});
    });
}

async function initialize() {
    data = await loadCSV('university_ranking.csv');

    // Extract categories dynamically
    if (data.length > 0) {
        const columns = Object.keys(data[0]);
        categories = columns.slice(2); // Assume categories start at index 2
    }

    displayFilters();
    setupRegionFilter(); // Set up the region filter
    displayRankings();
}

function displayFilters() {
    const table = document.getElementById('filterTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    // Filter categories to exclude "Continent" and any non-research fields
    const filteredCategories = categories.filter(category => category !== 'Continent');
    
    filteredCategories.forEach(category => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${category}</td>
            <td>
                <label class="switch">
                    <input id="${category}" type="checkbox" onclick="resetPageAndDisplayRankings()">
                    <span class="slider round"></span>
                </label>
            </td>`;
    });
}

function setupRegionFilter() {
    const regionFilter = document.getElementById('regionFilter');
    regionFilter.addEventListener('change', resetPageAndDisplayRankings); // Add listener for region change
}

function resetPageAndDisplayRankings() {
    currentPage = 1;
    selectedRegion = document.getElementById('regionFilter').value; // Get selected region
    displayRankings();
}

function displayRankings() {
    const calculatedScores = [];
    const seenUniversities = new Set(); // Set to track universities that have been added

    data.forEach(university => {
        const totalScore = categories.reduce((sum, category) => {
            const checkbox = document.getElementById(category);
            if (checkbox?.checked) {
                sum += getScore(university.University, category);
            }
            return sum;
        }, 0);

        // Apply region filtering: Make sure to handle case where Continent is empty or invalid
        if (selectedRegion !== 'all' && university.Continent) {
            // Ensure case-insensitive comparison and handle missing regions
            const continentMatch = university.Continent.trim().toLowerCase() === selectedRegion.toLowerCase();
            if (!continentMatch) {
                return; // Skip this university if it doesn't match the selected region
            }
        }

        if (!seenUniversities.has(university.University)) {
            calculatedScores.push({
                University: university.University,
                Continent: university.Continent || 'Unknown',
                Score: totalScore
            });
            seenUniversities.add(university.University);
        }
    });

    calculatedScores.sort((a, b) => b.Score - a.Score);

    displayPage(currentPage, calculatedScores);
    updatePageIndicator(currentPage, Math.ceil(calculatedScores.length / rowsPerPage));
}

function displayPage(page, data) {
    const table = document.getElementById('rankingTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    const start = (page - 1) * rowsPerPage;
    const end = page * rowsPerPage;

    data.slice(start, end).forEach((university, index) => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${start + index + 1}</td>
            <td>${university.University}</td>
            <td>${university.Score.toFixed(2)}</td>`;
    });
}

function updatePageIndicator(page, totalPages) {
    document.getElementById('pageIndicator').textContent = `Page ${page} of ${totalPages}`;
}

function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        displayRankings();
    }
}

function nextPage() {
    const totalPages = Math.ceil(data.length / rowsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        displayRankings();
    }
}

function getScore(universityName, categoryName) {
    const row = data.find(entry => entry.University === universityName);
    if (!row) return 0;

    const score = parseFloat(row[categoryName]);
    return isNaN(score) ? 0 : score;
}

function toggleAllCheckboxes() {
    const checkboxes = document.querySelectorAll('#filterTable input[type="checkbox"]');
    const allChecked = Array.from(checkboxes).every(checkbox => checkbox.checked);

    checkboxes.forEach(checkbox => checkbox.checked = !allChecked);
    document.getElementById('toggleAll').textContent = allChecked ? 'Select All' : 'Deselect All';

    resetPageAndDisplayRankings(); // Re-display rankings after toggle
}

// Initialize the application at the end of the script
initialize();