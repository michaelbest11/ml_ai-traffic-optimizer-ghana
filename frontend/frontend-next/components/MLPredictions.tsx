"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

type Prediction = {
  timestamp: string;
  congestion_level: string;
  recommended_action: string;
};

const samplePredictions: Record<string, Prediction[]> = {
  Accra: [
    {
      timestamp: "2025-07-07T08:00:00Z",
      congestion_level: "High",
      recommended_action: "Use alternate route via Ring Road",
    },
    {
      timestamp: "2025-07-07T09:00:00Z",
      congestion_level: "Moderate",
      recommended_action: "Expect delays on N1 Highway",
    },
  ],
  Kumasi: [
    {
      timestamp: "2025-07-07T08:00:00Z",
      congestion_level: "Low",
      recommended_action: "No delays expected",
    },
    {
      timestamp: "2025-07-07T09:00:00Z",
      congestion_level: "High",
      recommended_action: "Avoid Kejetia Market route",
    },
  ],
};

const MLPredictions = () => {
  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [selectedCity, setSelectedCity] = useState<"Accra" | "Kumasi">("Accra");
  const [loading, setLoading] = useState(false);

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/predictions/${selectedCity}`);
      if (response.data?.length > 0) {
        setPredictions(response.data);
      } else {
        // fallback to sample data
        setPredictions(samplePredictions[selectedCity]);
      }
    } catch (error) {
      console.warn("Using fallback predictions due to fetch error:", error);
      setPredictions(samplePredictions[selectedCity]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, [selectedCity]);

  return (
    <div className="p-4 border rounded shadow mt-4">
      <h2 className="text-xl font-semibold mb-2">ML Traffic Predictions</h2>

      <select
        className="mb-4 p-2 border rounded"
        value={selectedCity}
        onChange={(e) => setSelectedCity(e.target.value as "Accra" | "Kumasi")}
      >
        <option value="Accra">Accra</option>
        <option value="Kumasi">Kumasi</option>
      </select>

      {loading ? (
        <p>Loading predictions...</p>
      ) : predictions && predictions.length > 0 ? (
        <ul className="space-y-2">
          {predictions.map((prediction, index) => (
            <li key={index} className="border p-3 rounded bg-gray-50">
              <p>
                <strong>Time:</strong> {new Date(prediction.timestamp).toLocaleString()}
              </p>
              <p>
                <strong>Congestion:</strong> {prediction.congestion_level}
              </p>
              <p>
                <strong>Action:</strong> {prediction.recommended_action}
              </p>
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
