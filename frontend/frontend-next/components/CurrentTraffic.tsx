"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CurrentTraffic = () => {
  const [trafficData, setTrafficData] = useState<any>(null);
  const [selectedCity, setSelectedCity] = useState("Accra");
  const [loading, setLoading] = useState(false);

  const fetchTrafficData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/traffic/current/${selectedCity}`);
      setTrafficData(response.data);
    } catch (error) {
      console.error("Error fetching traffic data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrafficData();
  }, [selectedCity]);

  return (
    <div className="p-4 border rounded shadow">
      <h2 className="text-xl font-semibold mb-2">Current Traffic in {selectedCity}</h2>

      <select
        className="mb-4 p-2 border rounded"
        value={selectedCity}
        onChange={(e) => setSelectedCity(e.target.value)}
      >
        <option value="Accra">Accra</option>
        <option value="Kumasi">Kumasi</option>
      </select>

      {loading ? (
        <p>Loading traffic data...</p>
      ) : trafficData ? (
        <div>
          <p><strong>Congestion Level:</strong> {trafficData.congestion_level}</p>
          <p><strong>Average Speed:</strong> {trafficData.average_speed} km/h</p>
          <p><strong>Incidents:</strong> {trafficData.incidents?.length || 0}</p>
        </div>
      ) : (
        <p>No traffic data available.</p>
      )}
    </div>
  );
};

export default CurrentTraffic;
