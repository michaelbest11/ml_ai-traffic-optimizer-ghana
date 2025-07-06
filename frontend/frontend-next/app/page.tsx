"use client";
import React, { useState } from "react";
import TrafficDashboard from "../components/TrafficDashboard";
import RouteRecommendation from "../components/RouteRecommendation";
import MLPredictions from "../components/MLPredictions";
import MLAnalytics from "../components/MLAnalytics";
import CurrentTraffic from "../components/CurrentTraffic";
import "../app/globals.css"; // Adjust if using Tailwind or PostCSS

export function Page() {
  const [showPredictions, setShowPredictions] = useState(true);

  return (
    <main className="p-6">
      <TrafficDashboard />
      <RouteRecommendation />
      {showPredictions && <MLPredictions />}
      <MLAnalytics />
      <CurrentTraffic />
    </main>
  );
}

const tabs = [
  { id: 'dashboard', name: 'Traffic Dashboard', icon: '🚦' },
  { id: 'route', name: 'Route Optimization', icon: '🗺️' },
  { id: 'ml-predict', name: 'AI/ML Predictions', icon: '🤖' },
  { id: 'ml-analytics', name: 'ML Analytics', icon: '📊' },
  { id: 'live', name: 'Live Traffic', icon: '📍' }
];

export default function Home() {
  const [activeTab, setActiveTab] = useState("dashboard");

  return (
       <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">
            🚦 AI/ML Traffic Flow Optimizer
          </h1>
          <div className="text-sm text-gray-600">
            Ghana • Powered by Google AI & ML
          </div>
        </div>
      </header>

      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 flex space-x-8 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.icon} {tab.name}
            </button>
          ))}
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {activeTab === "dashboard" && <TrafficDashboard />}
        {activeTab === "route" && <RouteRecommendation />}
        {activeTab === "ml-predict" && <MLPredictions />}
        {activeTab === "ml-analytics" && <MLAnalytics />}
        {activeTab === "live" && <CurrentTraffic />}
      </main>

      <footer className="bg-gray-800 text-white py-4 mt-12 text-center">
        <p>© 2025 AI/ML Traffic Optimizer • Built for Ghana 🇬🇭</p>
      </footer>
    </div>
  );
};