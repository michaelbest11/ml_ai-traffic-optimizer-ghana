"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

type Prediction = {
  timestamp: string;
  city: string;
  congestion_level: string;
};

const MLPredictions = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [selectedCity, setSelectedCity] = useState("Accra");
  const [loading, setLoading] = useState(false);

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/predictions/ml/${selectedCity}`);
      setPredictions(response.data);
    } catch (error) {
      console.error("Error fetching ML predictions:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, [selectedCity]);

  return (
    <div className="traffic-card">
      <h2>ML Predictions for {selectedCity}</h2>
      <select value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)}>
        <option value="Accra">Accra</option>
        <option value="Kumasi">Kumasi</option>
      </select>

      {loading ? (
        <p>Loading predictions...</p>
      ) : predictions.length > 0 ? (
        <ul>
          {predictions.map((prediction: Prediction, index: number) => (
            <li key={index}>
              <strong>{prediction.timestamp}</strong>:{" "}
              <span>{prediction.congestion_level}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p>No predictions available.</p>
      )}
    </div>
  );
};

export default MLPredictions;
