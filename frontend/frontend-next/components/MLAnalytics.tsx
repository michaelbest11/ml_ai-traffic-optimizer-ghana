"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

type InsightData = {
  accuracy: number;
  top_bottlenecks: string[];
  flow_score: number;
};

const MLAnalytics = () => {
  const [insights, setInsights] = useState<InsightData | null>(null);
  const [selectedCity, setSelectedCity] = useState("Accra");
  const [loading, setLoading] = useState(false);

  const fetchMLInsights = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/analytics/ml-insights/${selectedCity}`);
      setInsights(response.data);
    } catch (error) {
      console.error("Error fetching ML insights:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMLInsights();
  }, [selectedCity]);

  return (
    <div className="traffic-card">
      <h2>ML Analytics for {selectedCity}</h2>
      <select value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)}>
        <option value="Accra">Accra</option>
        <option value="Kumasi">Kumasi</option>
      </select>

      {loading ? (
        <p>Loading insights...</p>
      ) : insights ? (
        <div>
          <p><strong>Prediction Accuracy:</strong> {insights.accuracy}%</p>
          <p><strong>Top Bottlenecks:</strong> {insights.top_bottlenecks.join(", ")}</p>
          <p><strong>Traffic Flow Score:</strong> {insights.flow_score}</p>
        </div>
      ) : (
        <p>No analytics data available.</p>
      )}
    </div>
  );
};

export default MLAnalytics;
