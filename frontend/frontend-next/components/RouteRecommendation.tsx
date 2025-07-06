"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

type Route = {
  from: string;
  to: string;
  estimated_time: string;
  suggested_route: string;
};

const RouteRecommendation = () => {
  const [routes, setRoutes] = useState<Route[]>([]);
  const [selectedCity, setSelectedCity] = useState("Accra");
  const [loading, setLoading] = useState(false);

  const fetchRoutes = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/routes/${selectedCity}`);
      setRoutes(response.data);
    } catch (error) {
      console.error("Error fetching routes:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRoutes();
  }, [selectedCity]);

  return (
    <div className="traffic-card">
      <h2>Recommended Routes - {selectedCity}</h2>
      <select value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)}>
        <option value="Accra">Accra</option>
        <option value="Kumasi">Kumasi</option>
      </select>
      {loading ? (
        <p>Loading route recommendations...</p>
      ) : routes.length > 0 ? (
        <ul>
          {routes.map((route, index) => (
            <li key={index}>
              <strong>{route.from} ➜ {route.to}</strong><br />
              Suggested: {route.suggested_route}<br />
              ETA: {route.estimated_time}
            </li>
          ))}
        </ul>
      ) : (
        <p>No routes found.</p>
      )}
    </div>
  );
};

export default RouteRecommendation;
