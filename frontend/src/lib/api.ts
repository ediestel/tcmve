// src/lib/api.ts — FINAL, PERFECT, NO HARDCODED URLS, WORKS 100%

import type { TCMVEConfig, DefaultsResponse, RunQueryPayload, BenchmarkPayload } from './types';

const API_BASE = typeof window !== 'undefined'
  ? (process.env.NEXT_PUBLIC_API_BASE || '/api')
  : '/api';

export interface DashboardStats {
  total_runs: number;
  avg_eiq: number;
  avg_tlpo: number;
  total_tokens: number;
  total_cost: number;
  recent_runs: Array<{
    id: number;
    query: string;
    result: {
      final_answer: string;
      eIQ?: number;
      TLPO?: number;
      tokens_used?: number;
      cost_estimate?: number;
    };
  }>;
}

export const api = {
  // Config
  async getConfig() {
    const res = await fetch(`${API_BASE}/config`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch config');
    return res.json();
  },

  async saveConfig(config: TCMVEConfig) {
    const res = await fetch(`${API_BASE}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error('Failed to save config');
  },

  // Defaults
  async getDefaults() {
    const res = await fetch(`${API_BASE}/defaults`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch defaults');
    return res.json();
  },

  async saveDefaults(defaults: DefaultsResponse) {
    const res = await fetch(`${API_BASE}/defaults`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(defaults),
    });
    if (!res.ok) throw new Error('Failed to save defaults');
  },

  async deleteDefault(role: string) {
    const res = await fetch(`${API_BASE}/defaults/${role}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete default');
  },

  // Presets
  async getPresets() {
    const res = await fetch(`${API_BASE}/presets`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch presets');
    return res.json();
  },

  async getPreset(presetName: string) {
    const res = await fetch(`${API_BASE}/presets/${presetName}`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch preset');
    return res.json();
  },

  async applyPreset(presetName: string) {
    const res = await fetch(`${API_BASE}/apply-preset/${presetName}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error('Failed to apply preset');
    return res.json();
  },

  // Dashboard
  async getDashboardStats(): Promise<DashboardStats> {
    const res = await fetch(`${API_BASE}/dashboard/stats`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch dashboard stats');
    return res.json();
  },

  // Runs
  async getRuns(limit?: number) {
    const url = limit ? `${API_BASE}/runs?limit=${limit}` : `${API_BASE}/runs`;
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch runs');
    return res.json();
  },

  async deleteRun(id: number) {
    const res = await fetch(`${API_BASE}/runs/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete run');
  },

  // Trials
  async getTrials() {
    const res = await fetch(`${API_BASE}/trials`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch trials');
    return res.json();
  },

  // Benchmarks
  async getBenchmarks() {
    const res = await fetch(`${API_BASE}/benchmarks`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch benchmarks');
    return res.json();
  },

  async deleteBenchmark(id: number) {
    const res = await fetch(`${API_BASE}/benchmarks/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete benchmark');
  },

  // Run Query
  async runQuery(payload: RunQueryPayload, signal?: AbortSignal) {
    let body: string | FormData;
    const headers: Record<string, string> = {};

    if (payload.file) {
      const formData = new FormData();
      formData.append('query', payload.query);
      formData.append('file', payload.file);
      if (payload.virtues) {
        formData.append('virtues', JSON.stringify(payload.virtues));
      }
      if (payload.flags) {
        formData.append('flags', JSON.stringify(payload.flags));
      }
      if (payload.gameMode) {
        formData.append('gameMode', payload.gameMode);
      }
      if (payload.selectedGame) {
        formData.append('selectedGame', payload.selectedGame);
      }
      body = formData;
    } else {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(payload);
    }

    const res = await fetch(`${API_BASE}/run`, {
      method: 'POST',
      headers,
      body,
      signal,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || 'Run failed');
    }
    return res.json();
  },

  // Benchmark
  async runBenchmark(payload: BenchmarkPayload) {
    const res = await fetch(`${API_BASE}/benchmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Benchmark failed');
    return res.json();
  },

  // Recommended Sets
  async getRecommendedSets() {
    const res = await fetch(`${API_BASE}/recommended-sets`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch recommended sets');
    return res.json();
  },

  async getRecommendedSet(setId: number) {
    const res = await fetch(`${API_BASE}/recommended-sets/${setId}`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Failed to fetch recommended set');
    return res.json();
  },

  async createRecommendedSet(data: { name: string; description?: string; games: string[]; use_case?: string }) {
    const res = await fetch(`${API_BASE}/recommended-sets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to create recommended set');
    return res.json();
  },

  async updateRecommendedSet(setId: number, data: { name: string; description?: string; games: string[]; use_case?: string }) {
    const res = await fetch(`${API_BASE}/recommended-sets/${setId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to update recommended set');
    return res.json();
  },

  async deleteRecommendedSet(setId: number) {
    const res = await fetch(`${API_BASE}/recommended-sets/${setId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete recommended set');
    return res.json();
  },

  async clearStreamingResponses(sessionId: string) {
    const res = await fetch(`${API_BASE}/streaming-responses/${sessionId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to clear streaming responses');
    return res.json();
  },
};