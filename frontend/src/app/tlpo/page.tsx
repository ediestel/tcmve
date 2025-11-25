// src/app/tlpo/page.tsx — FINAL, BEAUTIFUL, PRESERVES EVERYTHING
'use client';

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import ReactMarkdown from 'react-markdown';
import Sidebar from "@/components/Sidebar";
import type { TCMVEResult } from "@/lib/types";

interface Run {
  id: number;
  timestamp: string;
  query: string;
  description: string | null;
  final_answer: string;
  eiq: number | null;
  tlpo_scores: TCMVEResult['tlpo_scores'];
  tlpo_markup: string;
  converged?: boolean;
  rounds?: number;
}

export default function TLPO() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchRuns();
  }, []);

  const fetchRuns = async () => {
    try {
      const data = await api.getRuns(50);
      if (Array.isArray(data)) {
        setRuns(data);
      } else {
        console.error("API returned non-array:", data);
        setRuns([]);
      }
    } catch (error) {
      console.error("Fetch error:", error);
      setRuns([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 via-gray-100 to-gray-50 text-gray-900 flex">
      <Sidebar />

      {/* Main Content — YOUR ORIGINAL LAYOUT + NEW DATA */}
      <main className="flex-1 p-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-4xl font-bold text-gray-600 mb-8">TLPO Results</h1>

          {loading ? (
            <div className="text-center py-12 text-gray-500">Loading runs...</div>
          ) : runs.length === 0 ? (
            <div className="text-center py-12 text-gray-500">Database table empty or not connected</div>
          ) : (
            <div className="grid gap-6">
              {runs.map((run) => (
                <Card key={run.id} className="bg-white border-gray-300 shadow-sm">
                  <CardHeader>
                    <CardTitle className="text-xl text-gray-700">
                      {run.description || `Run ${run.id}`}
                      {run.eiq && (
                        <span className="ml-4 text-green-600 font-bold text-lg">
                          eIQ {Math.round(run.eiq)}
                        </span>
                      )}
                      {run.converged && (
                        <span className="ml-4 text-emerald-600 text-sm">✓ Converged in {run.rounds} rounds</span>
                      )}
                    </CardTitle>
                    <p className="text-sm text-gray-500">
                      {new Date(run.timestamp).toLocaleString()}
                    </p>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <strong>Query:</strong> <span className="text-gray-700">{run.query}</span>
                    </div>
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown>{run.final_answer}</ReactMarkdown>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6 text-sm">
                      <div>
                        <h3 className="font-semibold text-gray-600 mb-2">TLPO Scores</h3>
                        <pre className="bg-gray-50 p-3 rounded text-xs overflow-x-auto">
                          {JSON.stringify(run.tlpo_scores, null, 2)}
                        </pre>
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-600 mb-2">TLPO Markup (XML)</h3>
                        <pre className="bg-gray-50 p-3 rounded text-xs overflow-x-auto text-gray-600">
                          {run.tlpo_markup}
                        </pre>
                      </div>
                      <details className="mt-4">
                        <summary className="cursor-pointer text-sm font-semibold text-gray-600">Show full TCMVE trace (rounds, virtues, logs)</summary>
                        <pre className="mt-2 bg-gray-900 text-green-400 p-4 rounded overflow-auto max-h-96 text-xs">
                          {JSON.stringify(run, null, 2)}
                        </pre>
                      </details>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
  