"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";

const LoginPage = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("authority");
  const [error, setError] = useState("");
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!email || !password) {
      setError("Please enter both email and password.");
      return;
    }

    // Optional: call your backend here for real auth

    if (role === "authority") {
      router.push("/authority");
    } else {
      router.push("/driver");
    }
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100 px-4">
      <div className="bg-white p-8 shadow-lg rounded-md w-full max-w-md">
        <h2 className="text-2xl font-semibold mb-6 text-center">Login</h2>
        <form onSubmit={handleLogin} className="space-y-4">
          <input
            type="email"
            placeholder="Email"
            className="w-full px-4 py-2 border rounded-md"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            type="password"
            placeholder="Password"
            className="w-full px-4 py-2 border rounded-md"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <select
            className="w-full px-4 py-2 border rounded-md"
            value={role}
            onChange={(e) => setRole(e.target.value)}
          >
            <option value="authority">Traffic Authority</option>
            <option value="driver">Driver</option>
          </select>

          {error && <p className="text-red-500 text-sm">{error}</p>}

          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700"
          >
            Login
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;
