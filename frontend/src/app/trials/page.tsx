'use client';

/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import Sidebar from "@/components/Sidebar";

interface TrialResult {
  [key: string]: unknown; // Generic result structure
}

interface TrialData {
  filename: string;
  trial_type?: string;
  metadata?: Record<string, unknown>;
  analysis?: Record<string, unknown>;
  results?: TrialResult[];
  [key: string]: unknown; // Allow additional properties
}

export default function TrialsDashboard() {
  const [trials, setTrials] = useState<TrialData[]>([]);
  const [selectedTrial, setSelectedTrial] = useState<TrialData | null>(null);
  const [loading, setLoading] = useState(true);

  // Helper function to safely access analysis data
  const getAnalysisData = (trial: TrialData) => trial.analysis as any;

  useEffect(() => {
    const loadTrials = async () => {
      try {
        const data = await api.getTrials();
        setTrials(data.trials);
      } catch (error) {
        console.error("Failed to load trials", error);
      } finally {
        setLoading(false);
      }
    };
    loadTrials();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-linear-to-br from-gray-50 via-gray-100 to-gray-50 text-gray-900 flex">
        <Sidebar />
        <main className="flex-1 p-8">
          <div className="max-w-6xl mx-auto text-center py-32">
            <p className="text-xl text-gray-500">Loading ARCHER trials...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 via-gray-100 to-gray-50 text-gray-900 flex">
      <Sidebar />

      <main className="flex-1 p-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-4xl font-bold text-gray-600 mb-8">ARCHER Intelligence Enhancement Trials</h1>

          {!selectedTrial ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                {trials.map((trial) => (
                  <Card key={trial.filename} className="bg-white border-gray-300 cursor-pointer hover:shadow-lg transition-shadow"
                        onClick={() => setSelectedTrial(trial)}>
                    <CardHeader>
                      <CardTitle className="text-gray-600">
                        {trial.trial_type || 'Trial'} - {trial.filename}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-gray-500 mb-2">
                        {trial.metadata?.timestamp ? String(trial.metadata.timestamp) : 'Unknown date'}
                      </p>
                      {trial.analysis?.average_enhancement ? (
                        <p className="text-lg font-semibold text-green-600">
                          Avg Enhancement: {typeof trial.analysis.average_enhancement === 'number' ? (trial.analysis.average_enhancement * 100).toFixed(1) + '%' : 'N/A'}
                        </p>
                      ) : null}
                      {trial.metadata?.num_persons ? (
                        <p className="text-sm text-gray-600">
                          Sample: {String(trial.metadata.num_persons)} persons
                        </p>
                      ) : null}
                      {trial.metadata?.llm_provider ? (
                        <p className="text-sm text-gray-600">
                          LLM: {String(trial.metadata.llm_provider)}
                        </p>
                      ) : null}
                      <p className="text-sm text-gray-600">
                        Type: {trial.trial_type || 'Generic'}
                      </p>
                    </CardContent>
                  </Card>
                ))}
              </div>

              {trials.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-xl text-gray-500">No trials found. Run ARCHER trial to see results here.</p>
                </div>
              )}
            </>
          ) : (
            <div>
              <button
                onClick={() => setSelectedTrial(null)}
                className="mb-6 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
              >
                ← Back to Trials List
              </button>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <Card className="bg-white border-gray-300">
                  <CardHeader>
                    <CardTitle className="text-gray-600">Trial Overview</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p><strong>Trial Type:</strong> {selectedTrial.trial_type || 'Generic'}</p>
                    {selectedTrial.metadata?.num_persons ? (
                      <p><strong>Sample Size:</strong> {String(selectedTrial.metadata.num_persons)}</p>
                    ) : null}
                    {selectedTrial.metadata?.mean_biq ? (
                      <p><strong>Mean bIQ:</strong> {typeof selectedTrial.metadata.mean_biq === 'number' ? selectedTrial.metadata.mean_biq.toFixed(1) : String(selectedTrial.metadata.mean_biq)}</p>
                    ) : null}
                    {selectedTrial.metadata?.sigma_biq ? (
                      <p><strong>bIQ Std Dev:</strong> {typeof selectedTrial.metadata.sigma_biq === 'number' ? selectedTrial.metadata.sigma_biq.toFixed(1) : String(selectedTrial.metadata.sigma_biq)}</p>
                    ) : null}
                    {selectedTrial.metadata?.llm_provider ? (
                      <p><strong>LLM Provider:</strong> {String(selectedTrial.metadata.llm_provider)}</p>
                    ) : null}
                    {selectedTrial.analysis?.average_enhancement ? (
                      <p><strong>Avg Enhancement:</strong> {typeof selectedTrial.analysis.average_enhancement === 'number' ? (selectedTrial.analysis.average_enhancement * 100).toFixed(1) + '%' : String(selectedTrial.analysis.average_enhancement)}</p>
                    ) : null}
                    {selectedTrial.analysis?.biq_enhancement_correlation ? (
                      <p><strong>bIQ-Enhancement Correlation:</strong> {typeof selectedTrial.analysis.biq_enhancement_correlation === 'number' ? selectedTrial.analysis.biq_enhancement_correlation.toFixed(3) : String(selectedTrial.analysis.biq_enhancement_correlation)}</p>
                    ) : null}
                  </CardContent>
                </Card>

                {selectedTrial.analysis?.enhancement_patterns ? (
                  <Card className="bg-white border-gray-300">
                    <CardHeader>
                      <CardTitle className="text-gray-600">Enhancement Patterns</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="space-y-1">
                        {Object.entries(selectedTrial.analysis.enhancement_patterns as Record<string, unknown>).map(([pattern, value]) => (
                          <li key={pattern} className={value ? "text-green-600" : "text-gray-500"}>
                            {pattern.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}: {value ? "✓" : "✗"}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                ) : null}
              </div>

              {selectedTrial.analysis?.iq_group_analysis ? (
                <Card className="bg-white border-gray-300 mb-8">
                  <CardHeader>
                    <CardTitle className="text-gray-600">IQ Group Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left p-2">IQ Range</th>
                            <th className="text-left p-2">Count</th>
                            <th className="text-left p-2">Avg bIQ</th>
                            <th className="text-left p-2">Avg eIQ</th>
                            <th className="text-left p-2">Avg Enhancement</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(selectedTrial.analysis.iq_group_analysis as Record<string, any>).map(([range, data]) => (
                            <tr key={range} className="border-b">
                              <td className="p-2">{range}</td>
                              <td className="p-2">{data?.count || 'N/A'}</td>
                              <td className="p-2">{data?.avg_biq ? (typeof data.avg_biq === 'number' ? data.avg_biq.toFixed(1) : String(data.avg_biq)) : 'N/A'}</td>
                              <td className="p-2">{data?.avg_eiq ? (typeof data.avg_eiq === 'number' ? data.avg_eiq.toFixed(1) : String(data.avg_eiq)) : 'N/A'}</td>
                              <td className="p-2">{data?.avg_enhancement ? (typeof data.avg_enhancement === 'number' ? (data.avg_enhancement * 100).toFixed(1) + '%' : String(data.avg_enhancement)) : 'N/A'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              ) : null}

              {selectedTrial.analysis?.gardner_analysis ? (
                <Card className="bg-white border-gray-300">
                  <CardHeader>
                    <CardTitle className="text-gray-600">Intelligence Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(selectedTrial.analysis.gardner_analysis as Record<string, any>).map(([intelligence, data]) => (
                        <div key={intelligence} className="p-4 border rounded">
                          <h3 className="font-semibold text-gray-800 mb-2">{intelligence.replace('_', ' ')}</h3>
                          <p className="text-sm text-gray-600">Avg Enhancement: {data?.avg_enhancement ? (typeof data.avg_enhancement === 'number' ? (data.avg_enhancement * 100).toFixed(1) + '%' : String(data.avg_enhancement)) : 'N/A'}</p>
                          <p className="text-sm text-gray-600">Correlation: {data?.correlation ? (typeof data.correlation === 'number' ? data.correlation.toFixed(3) : String(data.correlation)) : 'N/A'}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ) : null}

              {/* Generic data display for any additional trial data */}
              <Card className="bg-white border-gray-300">
                <CardHeader>
                  <CardTitle className="text-gray-600">Trial Data</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="max-h-96 overflow-y-auto">
                    <pre className="text-xs text-gray-600 whitespace-pre-wrap">
                      {JSON.stringify(selectedTrial, null, 2)}
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}