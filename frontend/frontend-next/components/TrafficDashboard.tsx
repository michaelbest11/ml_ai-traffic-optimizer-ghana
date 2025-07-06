"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TrafficDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [selectedCity, setSelectedCity] = useState('Accra');
  const [loading, setLoading] = useState(false);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/dashboard/overview/${selectedCity}`);
      setDashboardData(response.data);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, [selectedCity]);


  return (
    <section className="traffic-grid">
      {/* Traffic Status Card: Accra */}
      <div className="traffic-card traffic-status-high">
        <h3>Accra: High Congestion</h3>
        <p>Estimated delay: <strong>15–20 minutes</strong></p>
        <span className="status-badge">Status: High</span>
      </div>

      {/* Traffic Status Card: Kumasi */}
      <div className="traffic-card traffic-status-medium">
        <h3>Kumasi: Moderate Traffic</h3>
        <p>Estimated delay: <strong>5–10 minutes</strong></p>
        <span className="status-badge">Status: Medium</span>
      </div>

      {/* Traffic Status Card: Cape Coast */}
      <div className="traffic-card traffic-status-low">
        <h3>Cape Coast: Free Flow</h3>
        <p>No significant delays reported</p>
        <span className="status-badge">Status: Low</span>
      </div>

      {/* Heatmap Simulation */}
      <div className="traffic-card heatmap-critical">
        <h3>Real-Time Heatmap: Critical Zone</h3>
        <p>Several bottlenecks detected</p>
        <span className="status-badge">Heatmap Level: Critical</span>
      </div>
    </section>
  );
};

export default TrafficDashboard;
