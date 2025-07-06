"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CurrentTraffic = () => {
  const [trafficData, setTrafficData] = useState(null);
  const [selectedCity, setSelectedCity] = useState('Accra');
  const [loading, setLoading] = useState(false);

  const fetchTrafficData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/traffic/current/${selectedCity}`);
      setTrafficData(response.data);
    } catch (error) {
      console.error('Error fetching traffic data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrafficData();
  }, [selectedCity]);

  return (
    // ...Same JSX from CurrentTraffic
  );
};

export default CurrentTraffic;
