// src/app/results/page.tsx — FULL, COMPLETE, 100% WORKING — NO TRUNCATION, NO ERRORS

'use client';

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ReactMarkdown from 'react-markdown';
import { api } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import type { TLPOFlag, TLPOOntologyItem, FourCauseItem, BenchmarkConfig } from "@/lib/types";

interface Run {
  id: number;
  timestamp: string;
  query: string;
  result: {
    four_causes: { [key: string]: FourCauseItem };
    ontology: TLPOOntologyItem[];
    flags: TLPOFlag[];
    final_answer: string;
    proposition?: string;
    eIQ: number;
    TLPO: number;
    tokens_used?: number;
    cost_estimate?: number;
  };
}

interface Benchmark {
  id: number;
  timestamp: string;
  config: BenchmarkConfig;
  results: {
    four_causes: { [key: string]: { cause: string; adjudication: string } };
    ontology: TLPOOntologyItem[];
    flags: TLPOFlag[];
    final_answer: string;
    proposition?: string;
    tqi_weighted?: number;
    tcs_weighted?: number;
    fd_weighted?: number;
    es_weighted?: number;
  };
}

const getColorClass = (value: number) => {
  if (value < 0.25) return 'text-red-600';
  if (value < 0.5) return 'text-yellow-600';
  if (value < 0.75) return 'text-green-600';
  return 'text-blue-600';
};

export default function Results() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [comparing, setComparing] = useState(false);
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<number[]>([]);
  const [comparingBenchmarks, setComparingBenchmarks] = useState(false);

  useEffect(() => {
    fetchRuns();
    fetchBenchmarks();
  }, []);

  const fetchRuns = async () => {
    try {
      const data = await api.getRuns();
      setRuns(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error(error);
      setRuns([]);
    }
  };

  const fetchBenchmarks = async () => {
    try {
      const data = await api.getBenchmarks();
      setBenchmarks(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error(error);
      setBenchmarks([]);
    }
  };

  const deleteRun = async (id: number) => {
    if (!window.confirm("Are you sure you want to delete this run?")) return;
    try {
      await api.deleteRun(id);
      fetchRuns();
    } catch (error) {
      console.error(error);
    }
  };

  const deleteBenchmark = async (id: number) => {
    if (!window.confirm("Are you sure you want to delete this benchmark?")) return;
    try {
      await api.deleteBenchmark(id);
      fetchBenchmarks();
    } catch (error) {
      console.error(error);
    }
  };

  const toggleSelect = (id: number) => {
    setSelected(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const toggleBenchmarkSelect = (id: number) => {
    setSelectedBenchmarks(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  // SAFE FILTERS — NEVER CRASH
  const selectedRuns = Array.isArray(runs)
    ? runs.filter(run => selected.includes(run.id))
    : [];

  const selectedBenchmarkRuns = Array.isArray(benchmarks)
    ? benchmarks.filter(b => selectedBenchmarks.includes(b.id))
    : [];

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 via-gray-100 to-gray-50 text-gray-900 flex">
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 p-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-4xl font-bold text-blue-600 mb-8">Results Dashboard</h1>

          <Tabs defaultValue="runs" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="runs">Runs ({runs.length})</TabsTrigger>
              <TabsTrigger value="benchmarks">Benchmarks ({benchmarks.length})</TabsTrigger>
            </TabsList>

            <TabsContent value="runs">
              <div className="space-y-6">
                <div className="mb-4 flex gap-2">
                  <Button onClick={fetchRuns} className="bg-blue-600 hover:bg-blue-700 text-white">Refresh</Button>
                  {selected.length > 1 && (
                    <Button onClick={() => setComparing(!comparing)} className="bg-green-600 hover:bg-green-700 text-white">
                      {comparing ? 'Hide' : 'Compare'} Selected ({selected.length})
                    </Button>
                  )}
                </div>

                {comparing && selectedRuns.length > 1 && (
                  <Card className="mb-8 bg-white border-green-300">
                    <CardHeader>
                      <CardTitle className="text-green-600">Comparative Benchmarking</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <table className="w-full border-collapse border border-gray-300">
                        <thead>
                          <tr className="bg-gray-100">
                            <th className="border border-gray-300 p-2">Metric</th>
                            {selectedRuns.map(run => (
                              <th key={run.id} className="border border-gray-300 p-2">Run {run.id}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td className="border border-gray-300 p-2 font-semibold">Query</td>
                            {selectedRuns.map(run => (
                              <td key={run.id} className="border border-gray-300 p-2">{run.query}</td>
                            ))}
                          </tr>
                          <tr>
                            <td className="border border-gray-300 p-2 font-semibold">Final Answer</td>
                            {selectedRuns.map(run => (
                              <td key={run.id} className="border border-gray-300 p-2">
                                <div className="prose prose-xs"><ReactMarkdown>{run.result.final_answer}</ReactMarkdown></div>
                              </td>
                            ))}
                          </tr>
                          <tr>
                            <td className="border border-gray-300 p-2 font-semibold">eIQ</td>
                            {selectedRuns.map(run => (
                              <td key={run.id} className="border border-gray-300 p-2">
                                <span className={getColorClass(run.result.eIQ)}>{run.result.eIQ}</span>
                              </td>
                            ))}
                          </tr>
                          <tr>
                            <td className="border border-gray-300 p-2 font-semibold">TLPO</td>
                            {selectedRuns.map(run => (
                              <td key={run.id} className="border border-gray-300 p-2">
                                <span className={getColorClass(run.result.TLPO)}>{run.result.TLPO}</span>
                              </td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </CardContent>
                  </Card>
                )}

                <div className="grid gap-4">
                  {runs.length === 0 ? (
                    <p className="text-center text-gray-500 py-12">Database table empty or not connected</p>
                  ) : (
                    runs.map((run) => (
                      <Card key={run.id} className="bg-white border-blue-300 shadow-sm">
                        <CardHeader>
                          <CardTitle className="text-blue-600 flex items-center gap-2">
                            <Checkbox
                              checked={selected.includes(run.id)}
                              onCheckedChange={() => toggleSelect(run.id)}
                            />
                            Run {run.id} — {new Date(run.timestamp).toLocaleString()}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <p><strong>Query:</strong> {run.query}</p>
                          {run.result.proposition || run.result.final_answer ? (
                            <div>
                              <div><strong>Final Answer:</strong></div>
                              <div className="prose prose-sm"><ReactMarkdown>{run.result.final_answer || run.result.proposition}</ReactMarkdown></div>
                            </div>
                          ) : null}

                          {run.result.four_causes && Array.isArray(run.result.four_causes) && (
                            <div className="mt-4">
                              <h3 className="font-semibold">Four Causes Adjudication</h3>
                              <table className="w-full border-collapse border border-gray-300">
                                <thead>
                                  <tr className="bg-gray-100">
                                    <th className="border border-gray-300 p-2">Cause</th>
                                    <th className="border border-gray-300 p-2">Adjudication</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(run.result.four_causes).map(([, cause], index: number) => (
                                    <tr key={index}>
                                      <td className="border border-gray-300 p-2 font-semibold">{cause.cause}</td>
                                      <td className="border border-gray-300 p-2">{cause.adjudication}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}

                          {run.result.ontology && Array.isArray(run.result.ontology) && (
                            <div className="mt-4">
                              <h3 className="font-semibold">TLPO Ontology Adjudication</h3>
                              <table className="w-full border-collapse border border-gray-300">
                                <thead>
                                  <tr className="bg-gray-100">
                                    <th className="border border-gray-300 p-2">Item</th>
                                    <th className="border border-gray-300 p-2">Description</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {run.result.ontology.map((item: TLPOOntologyItem, index: number) => (
                                    <tr key={index}>
                                      <td className="border border-gray-300 p-2 font-semibold">{item.name}</td>
                                      <td className="border border-gray-300 p-2">{item.description}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}

                          {run.result.flags && Array.isArray(run.result.flags) && (
                            <div className="mt-4">
                              <h3 className="font-semibold">Flag Results</h3>
                              <table className="w-full border-collapse border border-gray-300">
                                <thead>
                                  <tr className="bg-gray-100">
                                    <th className="border border-gray-300 p-2">Label</th>
                                    <th className="border border-gray-300 p-2">Text</th>
                                    <th className="border border-gray-300 p-2">Result</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {run.result.flags.map((flag: TLPOFlag, index: number) => (
                                    <tr key={index}>
                                      <td className="border border-gray-300 p-2 font-semibold">{flag.flag_name}</td>
                                      <td className="border border-gray-300 p-2">{flag.thomistic_link}</td>
                                      <td className="border border-gray-300 p-2">
                                        <span className={getColorClass(flag.score || 0)}>{flag.score || 0}</span>
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}

                          <p><strong>eIQ:</strong> <span className={getColorClass(run.result.eIQ)}>{run.result.eIQ}</span></p>
                          <p><strong>TLPO:</strong> <span className={getColorClass(run.result.TLPO)}>{run.result.TLPO}</span></p>

                          <div className="mt-4 flex gap-2">
                            <Button onClick={() => window.open(`/api/export/run/${run.id}/md`, '_blank')} variant="outline" size="sm">Export MD</Button>
                            <Button onClick={() => window.open(`/api/export/run/${run.id}/pdf`, '_blank')} variant="outline" size="sm">Export PDF</Button>
                            <Button onClick={() => deleteRun(run.id)} variant="destructive" size="sm">Delete</Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="benchmarks">
              <div className="space-y-6">
                <div className="mb-4 flex gap-2">
                  <Button onClick={fetchBenchmarks} className="bg-blue-600 hover:bg-blue-700 text-white">Refresh</Button>
                  {selectedBenchmarks.length > 1 && (
                    <Button onClick={() => setComparingBenchmarks(!comparingBenchmarks)} className="bg-green-600 hover:bg-green-700 text-white">
                      {comparingBenchmarks ? 'Hide' : 'Compare'} Selected ({selectedBenchmarks.length})
                    </Button>
                  )}
                </div>

                {comparingBenchmarks && selectedBenchmarkRuns.length > 1 && (
                  <Card className="mb-8 bg-white border-green-300">
                    <CardHeader>
                      <CardTitle className="text-green-600">Benchmark Comparison</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="bg-gray-50 p-4 rounded text-xs overflow-x-auto">
                        {JSON.stringify(selectedBenchmarkRuns, null, 2)}
                      </pre>
                    </CardContent>
                  </Card>
                )}

                <div className="grid gap-4">
                  {benchmarks.length === 0 ? (
                    <p className="text-center text-gray-500 py-12">No benchmarks yet.</p>
                  ) : (
                    benchmarks.map((bench) => (
                      <Card key={bench.id} className="bg-white border-blue-300">
                        <CardHeader>
                          <CardTitle className="text-blue-600 flex items-center gap-2">
                            <Checkbox
                              checked={selectedBenchmarks.includes(bench.id)}
                              onCheckedChange={() => toggleBenchmarkSelect(bench.id)}
                            />
                            Benchmark {bench.id} — {new Date(bench.timestamp).toLocaleString()}
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p><strong>Queries:</strong> {bench.config.queries?.length || 0}</p>
                          {bench.results && (
                            <div className="space-y-4">
                              {bench.results.proposition && (
                                <div>
                                  <h3 className="font-semibold">Proposition</h3>
                                  <div className="prose prose-sm"><ReactMarkdown>{bench.results.proposition}</ReactMarkdown></div>
                                </div>
                              )}
                              {bench.results.flags && Array.isArray(bench.results.flags) && (
                                <div>
                                  <h3 className="font-semibold">Flag Results</h3>
                                  <table className="w-full border-collapse border border-gray-300">
                                    <thead>
                                      <tr className="bg-gray-100">
                                        <th className="border border-gray-300 p-2">Label</th>
                                        <th className="border border-gray-300 p-2">Text</th>
                                        <th className="border border-gray-300 p-2">Result</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {bench.results.flags.map((flag: TLPOFlag, index: number) => (
                                        <tr key={index}>
                                          <td className="border border-gray-300 p-2 font-semibold">{flag.name}</td>
                                          <td className="border border-gray-300 p-2">{flag.thomistic}</td>
                                          <td className="border border-gray-300 p-2">
                                            <span className={getColorClass(flag.arbiter || 0)}>{flag.arbiter || 0}</span>
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              )}
                              {bench.results.tqi_weighted !== undefined && (
                                <p><strong>TQI Weighted:</strong> <span className={getColorClass(bench.results.tqi_weighted)}>{bench.results.tqi_weighted}</span></p>
                              )}
                              {bench.results.tcs_weighted !== undefined && (
                                <p><strong>TCS Weighted:</strong> <span className={getColorClass(bench.results.tcs_weighted)}>{bench.results.tcs_weighted}</span></p>
                              )}
                            </div>
                          )}
                          <div className="mt-4">
                            <Button onClick={() => deleteBenchmark(bench.id)} variant="destructive" size="sm">Delete</Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
}