// src/app/dashboard/page.tsx — FINAL, FULL, BEAUTIFUL, NO FETCH, USES api ONLY

'use client';

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import ReactMarkdown from 'react-markdown';
import { api } from "@/lib/api";
import Sidebar from "@/components/Sidebar";

// Define the exact shape from the backend
interface RecentRunFromBackend {
  id: number;
  query: string;
  result: {
    final_answer: string;
    eIQ?: number;
    TLPO?: number;
    tokens_used?: number;
    cost_estimate?: number;
  };
}

interface DashboardStats {
  total_runs: number;
  avg_eiq: number;
  avg_tlpo: number;
  total_tokens: number;
  total_cost: number;
  recent_runs: RecentRunFromBackend[];
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    total_runs: 0,
    avg_eiq: 0,
    avg_tlpo: 0,
    total_tokens: 0,
    total_cost: 0,
    recent_runs: []
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const data = await api.getDashboardStats();
        setStats(data);
      } catch (error) {
        console.error("Failed to load dashboard stats", error);
      } finally {
        setLoading(false);
      }
    };
    loadDashboard();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-linear-to-br from-gray-50 via-gray-100 to-gray-50 text-gray-900 flex">
        <Sidebar />
        <main className="flex-1 p-8">
          <div className="max-w-6xl mx-auto text-center py-32">
            <p className="text-xl text-gray-500">Loading the fire...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 via-gray-100 to-gray-50 text-gray-900 flex">
      <Sidebar />

      {/* MAIN CONTENT — YOUR ORIGINAL DESIGN + REAL DB DATA */}
      <main className="flex-1 p-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-4xl font-bold text-gray-600 mb-8">System Dashboard</h1>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-6 mb-8">
            <Card className="bg-white border-gray-300">
              <CardHeader>
                <CardTitle className="text-gray-600">Total Runs</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-4xl font-bold text-gray-800">{stats.total_runs}</p>
              </CardContent>
            </Card>
            <Card className="bg-white border-gray-300">
              <CardHeader>
                <CardTitle className="text-gray-600">Avg eIQ</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-4xl font-bold text-green-600">{stats.avg_eiq}</p>
              </CardContent>
            </Card>
            <Card className="bg-white border-gray-300">
              <CardHeader>
                <CardTitle className="text-gray-600">Avg TLPO</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-4xl font-bold text-blue-600">{stats.avg_tlpo}</p>
              </CardContent>
            </Card>
            <Card className="bg-white border-gray-300">
              <CardHeader>
                <CardTitle className="text-gray-600">Est. Tokens</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold text-gray-800">{stats.total_tokens.toLocaleString()}</p>
              </CardContent>
            </Card>
            <Card className="bg-white border-gray-300">
              <CardHeader>
                <CardTitle className="text-gray-600">Est. Cost</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold text-red-600">${stats.total_cost}</p>
              </CardContent>
            </Card>
          </div>

          {/* Cost Note */}
          <Card className="bg-white border-gray-300 mb-8">
            <CardHeader>
              <CardTitle className="text-gray-600">Cost Calculator (Est.)</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg">
                Approximate total cost: <span className="font-bold text-green-600">${stats.total_cost}</span> 
                {' '} (based on OpenAI GPT-4 rates)
              </p>
              <p className="text-sm text-gray-600 mt-2">
                Note: Actual costs may vary based on selected LLM provider and usage patterns.
              </p>
            </CardContent>
          </Card>

          {/* Recent Runs */}
          <div>
            <h2 className="text-2xl font-bold text-gray-600 mb-6">Recent Runs</h2>
            <div className="grid gap-6">
              {stats.recent_runs.length === 0 ? (
                <p className="text-center text-gray-500 py-12">No runs yet. The fire awaits.</p>
              ) : (
                stats.recent_runs.map((run) => (
                  <Card key={run.id} className="bg-white border-gray-300 shadow-sm">
                    <CardContent className="p-6">
                      <p className="font-semibold text-lg text-gray-800 mb-2">{run.query}</p>
                      <div className="prose prose-sm text-gray-700 mb-4">
                        <ReactMarkdown>{run.result.final_answer}</ReactMarkdown>
                      </div>
                      <div className="flex flex-wrap gap-4 text-sm text-gray-600">
                        <span>eIQ: <strong className="text-green-600">{run.result.eIQ || '—'}</strong></span>
                        <span>TLPO: <strong className="text-blue-600">{run.result.TLPO || '—'}</strong></span>
                        <span>Tokens: <strong>{run.result.tokens_used || 0}</strong></span>
                        <span>Cost: <strong className="text-red-600">${run.result.cost_estimate || 0}</strong></span>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}