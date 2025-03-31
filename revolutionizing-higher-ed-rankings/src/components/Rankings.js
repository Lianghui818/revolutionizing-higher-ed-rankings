// components/Rankings.js
import React, { useState, useEffect } from 'react';

function Rankings() {
  const [data, setData] = useState([]);
  const [categories, setCategories] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedRegion, setSelectedRegion] = useState('all');
  const [selectedCategories, setSelectedCategories] = useState({});
  const [calculatedScores, setCalculatedScores] = useState([]);
  const [toggleAllState, setToggleAllState] = useState(true);
  
  const rowsPerPage = 25;

  useEffect(() => {
    // Initial data loading
    initialize();
  }, []);

  useEffect(() => {
    // Recalculate rankings when filters or page changes
    displayRankings();
  }, [currentPage, selectedRegion, selectedCategories, data]);

  const loadCSV = async (filePath) => {
    try {
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
    } catch (error) {
      console.error("Error loading CSV:", error);
      return [];
    }
  };

  const initialize = async () => {
    const loadedData = await loadCSV('university_ranking.csv');
    setData(loadedData);

    // Extract categories dynamically
    if (loadedData.length > 0) {
      const columns = Object.keys(loadedData[0]);
      const extractedCategories = columns.slice(2); // Assume categories start at index 2
      setCategories(extractedCategories);
      
      // Initialize all categories as unchecked
      const categoryState = {};
      extractedCategories.forEach(category => {
        categoryState[category] = false;
      });
      setSelectedCategories(categoryState);
    }
  };

  const handleCategoryChange = (category) => {
    setSelectedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
    setCurrentPage(1); // Reset to first page when filters change
  };

  const handleRegionChange = (e) => {
    setSelectedRegion(e.target.value);
    setCurrentPage(1); // Reset to first page when region changes
  };

  const toggleAllCheckboxes = () => {
    const newState = !toggleAllState;
    const updatedCategories = {};
    
    categories.forEach(category => {
      if (category !== 'Continent') {
        updatedCategories[category] = newState;
      }
    });
    
    setSelectedCategories(updatedCategories);
    setToggleAllState(newState);
    setCurrentPage(1);
  };

  const displayRankings = () => {
    const scores = [];
    const seenUniversities = new Set();

    data.forEach(university => {
      const totalScore = categories.reduce((sum, category) => {
        if (selectedCategories[category]) {
          sum += getScore(university.University, category);
        }
        return sum;
      }, 0);

      // Apply region filtering
      if (selectedRegion !== 'all' && university.Continent) {
        const continentMatch = university.Continent.trim().toLowerCase() === selectedRegion.toLowerCase();
        if (!continentMatch) {
          return; // Skip this university if it doesn't match the selected region
        }
      }

      if (!seenUniversities.has(university.University)) {
        scores.push({
          University: university.University,
          Continent: university.Continent || 'Unknown',
          Score: totalScore
        });
        seenUniversities.add(university.University);
      }
    });

    scores.sort((a, b) => b.Score - a.Score);
    setCalculatedScores(scores);
  };

  const getScore = (universityName, categoryName) => {
    const row = data.find(entry => entry.University === universityName);
    if (!row) return 0;

    const score = parseFloat(row[categoryName]);
    return isNaN(score) ? 0 : score;
  };

  const prevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const nextPage = () => {
    const totalPages = Math.ceil(calculatedScores.length / rowsPerPage);
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    }
  };

  const filteredCategories = categories.filter(category => category !== 'Continent');
  const totalPages = Math.ceil(calculatedScores.length / rowsPerPage);
  const start = (currentPage - 1) * rowsPerPage;
  const visibleData = calculatedScores.slice(start, start + rowsPerPage);

  return (
    <div className="content">
      <div className="filter-container">
        <h2>Filters</h2>
        
        <div className="field-container">
          <label>Region: </label>
          <select 
            className="filter-select"
            value={selectedRegion} 
            onChange={handleRegionChange}
          >
            <option value="all">All Regions</option>
            <option value="north america">North America</option>
            <option value="europe">Europe</option>
            <option value="asia">Asia</option>
            <option value="oceania">Oceania</option>
            <option value="south america">South America</option>
            <option value="africa">Africa</option>
          </select>
        </div>

        <div className="button-container">
          <button 
            className="toggle-button" 
            onClick={toggleAllCheckboxes}
          >
            {toggleAllState ? 'Select All' : 'Deselect All'}
          </button>
        </div>
        
        <h3 className="align-left">Categories</h3>
        {filteredCategories.map(category => (
          <div key={category} className="field-container">
            <span>{category}</span>
            <label className="switch">
              <input 
                type="checkbox" 
                checked={selectedCategories[category] || false}
                onChange={() => handleCategoryChange(category)}
              />
              <span className="slider"></span>
            </label>
          </div>
        ))}
      </div>

      <div className="table-container">
        <h2>University Rankings</h2>
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>University</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {visibleData.map((university, index) => (
              <tr key={university.University}>
                <td>{start + index + 1}</td>
                <td>{university.University}</td>
                <td>{university.Score.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        
        <div className="pagination-container">
          <button onClick={prevPage} disabled={currentPage === 1}>Previous</button>
          <span id="pageIndicator">Page {currentPage} of {totalPages || 1}</span>
          <button onClick={nextPage} disabled={currentPage >= totalPages}>Next</button>
        </div>
      </div>
    </div>
  );
}

export default Rankings;