db = db.getSiblingDB('traffic_db');

db.current_traffic.insertMany([
  {
    city: "Accra",
    congestion_level: "High",
    speed_avg_kph: 28,
    last_updated: new Date()
  },
  {
    city: "Kumasi",
    congestion_level: "Moderate",
    speed_avg_kph: 35,
    last_updated: new Date()
  }
]);

db.ml_predictions.insertMany([
  {
    city: "Accra",
    prediction_time: new Date(),
    traffic_volume: 520,
    congestion_score: 0.8
  },
  {
    city: "Kumasi",
    prediction_time: new Date(),
    traffic_volume: 410,
    congestion_score: 0.6
  }
]);
