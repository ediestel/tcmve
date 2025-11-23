// API Response Types
export interface TLPOFlag {
  flag_name?: string;
  name?: string;
  thomistic_link?: string;
  thomistic?: string;
  score?: number;
  arbiter?: number;
  generator?: string;
  verifier?: string;
}

export interface TLPOOntologyItem {
  name: string;
  description: string;
  score?: number;
}

export interface FourCauseItem {
  cause: string;
  adjudication: string;
}

export interface FourCauses {
  [key: string]: FourCauseItem;
}

export interface GameResult {
  game: string;
  nash_equilibrium: string;
  eIQ_boost: number;
  payoffs?: { generator: number; verifier: number };
  generator_move?: string;
  verifier_move?: string;
  virtue_adjustments?: {
    generator: Record<string, number>;
    verifier: Record<string, number>;
  };
}

export interface RoundData {
  round: number;
  generator_input: string;
  proposition: string;
  verifier_input: string;
  refutation: string;
  game_applied?: string;
  games_applied?: string[];
  game_result?: GameResult;
  game_results?: GameResult[];
  game_error?: string;
}

export interface TCMVEResult {
  query: string;
  final_answer: string;
  converged: boolean;
  rounds: number;
  history: RoundData[];
  tlpo_scores: {
    flags: Record<string, number>;
    tqi: number;
    tcs: number;
  };
  tlpo_markup: string;
  metrics: {
    generator: { tqi: number; tcs: number; fd: number; es: number };
    verifier: { tqi: number; tcs: number; fd: number; es: number };
    arbiter: { tqi: number; tcs: number; fd: number; es: number };
    weighted_tqi: number;
    weighted_tcs: number;
  };
  eIQ?: number;
  TQI?: number;
  TCS?: number;
  tokens_used?: number;
  cost_estimate?: number;
  description?: string;
}

export interface BenchmarkResult {
  id: string;
  query: string;
  timestamp: string;
  result: {
    four_causes: FourCauses;
    ontology: TLPOOntologyItem[];
    flags: TLPOFlag[];
    final_answer: string;
  };
  config: {
    virtues: {
      generator: number[];
      verifier: number[];
      arbiter: number[];
    };
    game_mode: string;
  };
}

export interface BenchmarkRun {
  id: string;
  query: string;
  timestamp: string;
  result: BenchmarkResult['result'];
  config: BenchmarkResult['config'];
}

export interface PresetData {
  name: string;
  description: string;
  generator: Record<string, number>;
  verifier: Record<string, number>;
  arbiter: Record<string, number>;
  recommended_games: string[];
  use_case: string;
}

export interface Virtues {
  P: number;
  J: number;
  F: number;
  T: number;
  V: number;
  L: number;
  H: number;
  Ω: number;
}

export interface LLMProvider {
  generatorProvider: 'openai' | 'anthropic' | 'xai';
  verifierProvider: 'openai' | 'anthropic' | 'xai';
  arbiterProvider: 'openai' | 'anthropic' | 'xai';
}

export interface DefaultsResponse {
  generator: Virtues;
  verifier: Virtues;
  arbiter: Virtues;
}

export interface AvailablePresets {
  [key: string]: string; // preset name -> description
}

export interface TCMVEConfig {
  virtues: {
    generator: number[];
    verifier: number[];
    arbiter: number[];
  };
  flags: TCMVEFlags;
}

export interface TCMVEFlags {
  maritalFreedom: boolean;
  viceCheck: boolean;
  selfRefine: boolean;
  tlpoFull: boolean;
  noXml: boolean;
  sevenDomains: boolean;
  virtuesIndependent: boolean;
  useGenerator: boolean;
  useVerifier: boolean;
  useArbiter: boolean;
  arbiter_only: boolean;
  maxRounds: number;
  nashMode: 'on' | 'off' | 'auto';
  eiqLevel: number;
  simulatedPersons: number;
  biqDistribution: 'gaussian';
  meanBiq: number;
  sigmaBiq: number;
  output: string;
  generatorProvider: 'openai' | 'anthropic' | 'xai' | 'ollama';
  verifierProvider: 'openai' | 'anthropic' | 'xai' | 'ollama';
  arbiterProvider: 'openai' | 'anthropic' | 'xai' | 'ollama';
  streammode?: string;
}

export interface RunQueryPayload {
  query: string;
  file?: File;
  virtues?: {
    generator: number[];
    verifier: number[];
    arbiter: number[];
  };
  flags?: Partial<TCMVEFlags>;
  gameMode?: 'all' | 'separate' | 'recommended_set' | 'arbiterOnly';
  selectedGame?: string;
  session_id?: string;
}

export interface BenchmarkConfig {
  queries?: string[];
  virtues?: {
    generator: number[];
    verifier: number[];
    arbiter: number[];
  };
  game_mode?: string;
  latency_runs?: number;
  stability_cases?: Array<{
    base_query: string;
    contradictory_queries: string[];
  }>;
  adversarial_prompts?: string[];
  [key: string]: unknown; // Allow additional properties
}

export interface BenchmarkPayload {
  config: string; // JSON string
}