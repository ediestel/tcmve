'use client';

import { useState, useEffect, useRef, useCallback } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Upload } from "lucide-react";
import { VirtueSlider } from "@/components/ui/virtue-slider"
import { toast } from "sonner";
import ReactMarkdown from 'react-markdown';
import { api } from "@/lib/api";
import Sidebar from "@/components/Sidebar";

import type {
  TCMVEResult,
  DefaultsResponse,
  AvailablePresets,
  TCMVEFlags,
  RunQueryPayload
} from "@/lib/types";

const virtues = [
  "Ω Humility",
  "P Prudence",
  "J Justice",
  "F Fortitude",
  "T Temperance",
  "V Faith",
  "L Love",
  "H Hope"
] as const;

export default function Dashboard() {
  const [query, setQuery] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [lastResult, setLastResult] = useState<TCMVEResult | null>(null);
  const [defaults, setDefaults] = useState<DefaultsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [liveAnswer, setLiveAnswer] = useState("");

// ←←← NEW STATES – put them exactly here
  const [showUploadModal, setShowUploadModal] = useState(false); // controls the new upload modal
  // →→→ end of new states

  const abortControllerRef = useRef<AbortController | null>(null);
  const logsDivRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [config, defaults, presets, sets] = await Promise.all([
          api.getConfig(),
          api.getDefaults(),
          api.getPresets(),
          api.getRecommendedSets(),
        ]);
        if (config.virtues) {
          if (config.virtues.generator) setGenerator(config.virtues.generator);
          if (config.virtues.verifier) setVerifier(config.virtues.verifier);
          if (config.virtues.arbiter) setArbiter(config.virtues.arbiter);
        }
        if (config.flags) {
          setFlags(prev => ({ ...prev, ...config.flags }));
        }
        setDefaults(defaults);
        setAvailablePresets(presets);
        setAvailableSets(sets.sets);
      } catch (error) {
        console.error(error);
      }
    };
    load();
  }, []);




  // 3 personas × 8 virtues
  const [generator, setGenerator] = useState<number[]>([0.8, 0.8, 0.75, 0.65, 0.85, 0.72, 0.85, 0.89]);
  const [verifier, setVerifier] = useState<number[]>([0.9, 0.9, 0.95, 0.8, 0.9, 0.65, 0.9, 0.95]);
  const [arbiter, setArbiter] = useState<number[]>([0.85, 0.85, 0.8, 0.9, 0.85, 0.85, 0.8, 0.85]);

  const loadPreset = (role: string) => {
    if (defaults && defaults[role as keyof DefaultsResponse]) {
      const vals = Object.values(defaults[role as keyof DefaultsResponse]) as number[];
      if (role === 'generator') setGenerator(vals);
      else if (role === 'verifier') setVerifier(vals);
      else if (role.startsWith('arbiter')) setArbiter(vals);
    }
  };

  const applyVirtuePreset = async (presetName: string) => {
    try {
      // Get full preset details including recommended games
      const presetDetails = await api.getPreset(presetName);
      const result = await api.applyPreset(presetName);

      // Update local state with the preset values
      const preset = result.virtue_vectors;
      setGenerator(Object.values(preset.generator) as number[]);
      setVerifier(Object.values(preset.verifier) as number[]);
      setArbiter(Object.values(preset.arbiter) as number[]);
      setCurrentPreset(presetName);

      // Set recommended games (but don't auto-switch mode)
      const games = presetDetails.recommended_games || [];
      setRecommendedGames(games);

      toast.success(`Applied ${presetName.replace('_', ' ')} preset with ${games.length} recommended games`);
    } catch (error) {
      toast.error(`Failed to apply ${presetName} preset`);
      console.error(error);
    }
  };

  const arbiterPresets = defaults ? Object.keys(defaults).filter(key => key.startsWith('arbiter')) : [];
  const [selectedArbiterPreset, setSelectedArbiterPreset] = useState('arbiter');

  // Virtue presets state
  const [availablePresets, setAvailablePresets] = useState<AvailablePresets>({});
  const [currentPreset, setCurrentPreset] = useState<string>('');
  const [recommendedGames, setRecommendedGames] = useState<string[]>([]);

  // Recommended sets state
  const [availableSets, setAvailableSets] = useState<Array<{id: number, name: string, description: string, games: string[], use_case: string}>>([]);
  const [selectedSetId, setSelectedSetId] = useState<number | null>(null);

  // Flags
  const [flags, setFlags] = useState<TCMVEFlags>({
    maritalFreedom: false,
    viceCheck: true,
    selfRefine: true,
    tlpoFull: false,
    noXml: false,
    sevenDomains: true,
    virtuesIndependent: true,
    useGenerator: true,
    useVerifier: true,
    useArbiter: true,
    arbiter_only: false,
    maxRounds: 5,
    nashMode: 'auto' as 'on' | 'off' | 'auto',
    eiqLevel: 10,
    simulatedPersons: 24,
    biqDistribution: 'gaussian' as const,
    meanBiq: 100,
    sigmaBiq: 15,
    output: 'result',
    generatorProvider: 'openai' as 'openai' | 'anthropic' | 'xai' | 'ollama',
    verifierProvider: 'anthropic' as 'openai' | 'anthropic' | 'xai' | 'ollama',
    arbiterProvider: 'xai' as 'openai' | 'anthropic' | 'xai' | 'ollama',
    streammode: 'none',
  });

  const saveConfig = useCallback(() => {
    const config = {
      virtues: { generator, verifier, arbiter },
      flags,
    };
    api.saveConfig(config)
      .then(() => toast.success("Configuration saved"))
      .catch(() => toast.error("Failed to save configuration"));
  }, [generator, verifier, arbiter, flags]);

  const [gameMode, setGameMode] = useState<'all' | 'separate' | 'recommended_set' | 'arbiterOnly'>('all');
  const [selectedGame, setSelectedGame] = useState<string>('');
  const [advancedParams, setAdvancedParams] = useState(false);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      saveConfig();
    }, 1000); // Debounce 1 second
    return () => clearTimeout(timeoutId);
  }, [saveConfig]);

  useEffect(() => {
    if (logsDivRef.current) {
      logsDivRef.current.scrollTop = logsDivRef.current.scrollHeight;
    }
  }, [logs]);

  const calculateV = (values: number[]) => {
    const product = values.reduce((acc, v) => acc * v, 1);
    return (product * 1000).toFixed(3);
  };

  const V = Math.min(
    Number(calculateV(generator)),
    Number(calculateV(verifier)),
    Number(calculateV(arbiter))
  ).toFixed(3);

  const handleRun = async () => {
    setLoading(true);
    setLogs([]);
    setLiveAnswer("");

    // Generate unique session ID for this run
    const sessionId = 'test_session_1732300000000'; // Fixed ID for testing streaming

    const payload: RunQueryPayload = {
      query,
      ...(file && { file }),
      virtues: { generator, verifier, arbiter },
      flags,
      gameMode,
      selectedGame,
      session_id: sessionId,
    };

    // Create abort controller
    abortControllerRef.current = new AbortController();

    try {
      const result = await api.runQuery(payload, abortControllerRef.current.signal);
      setLastResult(result);
      toast.success(`Truth delivered – eIQ ${result.eIQ || 'N/A'}`);
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        toast.info("Query cancelled");
      } else {
        toast.error("Error running engine");
        console.error("Run error:", error);
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  }

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }


  useEffect(() => {
  console.log("FLAGS CHANGED:", {
    generator: flags.generatorProvider,
    verifier: flags.verifierProvider,
    arbiter: flags.arbiterProvider,
  });
}, [flags]);

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 via-gray-100 to-gray-50 text-gray-900 flex">
      <Sidebar />
      {/* Main Content */}
      <main className="flex-1 p-4">
        <div className="max-w-7xl mx-auto">
        <p className="text-center text-sm mb-8 text-blue-600">
          @ECKHART_DIESTEL — eIQ 2,441 — TQI 1.000 — V = {V}
        </p>

        <Tabs defaultValue="query" className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-8 bg-gray-200">
            <TabsTrigger value="query">Query & File</TabsTrigger>
            <TabsTrigger value="virtues">Virtue Organ</TabsTrigger>
            <TabsTrigger value="flags">Flags & Settings</TabsTrigger>
          </TabsList>

          {/* ==================== TAB 1 – Query & File ==================== */}
          <TabsContent value="query" className="space-y-6">
            <Card className="bg-white border-blue-300">
              <CardContent className="space-y-6">
                <div>
                  <Textarea
                    id="query"
                    placeholder="nTGT-Ω …"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="min-h-48 bg-gray-50 border-blue-300 text-gray-900"
                  />
                </div>

                <div className="text-center mt-6 space-y-4">
                  <div className="flex justify-center space-x-4">
                    <Button className="bg-gray-800 hover:bg-gray-700 text-white text-lg px-8 py-4" onClick={handleRun} disabled={loading}>
                      {loading ? "RUNNING..." : "RUN TRUTH ENGINE"}
                    </Button>
                    {loading && (
                      <Button className="bg-gray-400 hover:bg-gray-500 text-gray-800 text-lg px-6 py-4" onClick={handleCancel}>
                        CANCEL
                      </Button>
                    )}
                  </div>
                </div>






                <div className="text-center">
                  <Button variant="outline" onClick={() => setShowUploadModal(true)}>
                    <Upload className="mr-2 h-4 w-4" />
                    Upload PDF / TXT
                  </Button>
                  {file && <p className="mt-2 text-sm text-gray-600">Selected: {file.name}</p>}
                </div>
                {(loading || logs.length > 0) && (
                  <div className="mt-6">
                    <Card className="bg-white border-blue-300">
                      <CardHeader>
                        <CardTitle className="text-blue-600">Live Logs</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div ref={logsDivRef} className="h-64 overflow-y-auto bg-gray-50 p-4 font-mono text-sm">
                          {logs.map((log, i) => (
                            <div key={i}>{log}</div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
                {loading && liveAnswer && (
                  <div className="mt-8 p-8 bg-black text-green-400 font-mono text-xl rounded-lg overflow-hidden">
                    <div className="animate-pulse">Streaming from the Arbiter...</div>
                    <div className="mt-4 whitespace-pre-wrap wrap-break-words">
                      {liveAnswer}
                      <span className="animate-pulse">|</span>
                    </div>
                  </div>
                )}
                {lastResult && (
                  <div className="mt-6">
                    <Card className="bg-white border-green-300">
                      <CardHeader>
                        <CardTitle className="text-green-600">Response</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div><strong>Final Answer:</strong></div>
                        <div className="prose prose-sm max-h-96 overflow-y-auto"><ReactMarkdown>{lastResult.final_answer}</ReactMarkdown></div>
                        <div className="mt-4 flex gap-2">
                          <Button onClick={() => navigator.clipboard.writeText(lastResult.final_answer)} variant="outline">Copy</Button>
                          <Button onClick={() => toast.success("Response already saved to database")} variant="outline">Saved</Button>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* ==================== TAB 2 – Virtue Organ (COMPLETE) ==================== */}
          {/* ==================== TAB 2 – Virtue Organ (COMPLETE) ==================== */}
<TabsContent value="virtues">
  <div className="mb-6">
    <h2 className="text-xl font-semibold mb-3">Domain-Specific Virtue Presets</h2>
    <p className="text-sm text-gray-600 mb-4">
      Click a preset to apply optimized virtue configurations for specific ethical domains.
      {currentPreset && <span className="ml-2 font-medium text-blue-600">Current: {currentPreset.replace('_', ' ')}</span>}
    </p>

    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
      {Object.entries(availablePresets).map(([presetName, description]) => (
        <Button
          key={presetName}
          onClick={() => applyVirtuePreset(presetName)}
          variant={currentPreset === presetName ? "default" : "outline"}
          className={`text-xs p-2 h-auto ${
            currentPreset === presetName
              ? 'bg-blue-600 hover:bg-blue-700 text-white'
              : 'hover:bg-blue-50 border-blue-200'
          }`}
          title={description as string}
        >
          <div className="text-center">
            <div className="font-medium capitalize">
              {presetName.replace('_', ' ')}
            </div>
            <div className="text-xs opacity-75 mt-1">
              {presetName === 'healthcare_ethics' && '🏥'}
              {presetName === 'autonomous_vehicles' && '🚗'}
              {presetName === 'financial_risk' && '💰'}
              {presetName === 'legal_justice' && '⚖️'}
              {presetName === 'environmental_policy' && '🌱'}
              {presetName === 'academic_integrity' && '🎓'}
              {presetName === 'bauingenieur' && '🏗️'}
              {presetName === 'psychotherapy_cbt' && '🧠'}
            </div>
          </div>
        </Button>
      ))}
    </div>

    {recommendedGames.length > 0 && (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
        <p className="text-sm text-blue-800">
          <strong>🎮 Auto-selected Games:</strong> {recommendedGames.join(', ')} for {currentPreset.replace('_', ' ')} analysis
        </p>
      </div>
    )}

    <div className="border-t pt-4">
      <h3 className="text-sm font-medium mb-2">Legacy Presets</h3>
      <div className="flex space-x-2">
        <Button onClick={() => loadPreset('generator')} variant="outline" size="sm">Generator Defaults</Button>
        <Button onClick={() => loadPreset('verifier')} variant="outline" size="sm">Verifier Defaults</Button>
        <select
          value={selectedArbiterPreset}
          onChange={(e) => { setSelectedArbiterPreset(e.target.value); loadPreset(e.target.value); }}
          className="p-1 border rounded text-sm"
        >
          {arbiterPresets.map(preset => (
            <option key={preset} value={preset}>{preset.replace('arbiter', 'Arbiter').replace('_', ' ')}</option>
          ))}
        </select>
      </div>
    </div>
  </div>
  <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

    {/* ================================================== */}
    {/* =================== GENERATOR ==================== */}
    {/* ================================================== */}
    <Card className="bg-white border-yellow-400">
      <CardHeader className="text-center">
        <CardTitle className="text-2xl text-yellow-600">GENERATOR</CardTitle>
        <p className="text-yellow-500">Creative Fire</p>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-4 gap-6">
          {virtues.map((v, i) => (
            <div key={v} className="flex flex-col items-center gap-1">

              {/* virtue label */}
              <div className="h-4 flex items-center justify-center">
                <span className="text-xs text-yellow-600 text-center">{v}</span>
              </div>

              {/* fader */}
              <VirtueSlider
                value={[generator[i]]}
                onValueChange={(val) => {
                  const newG = [...generator];
                  newG[i] = val[0];
                  setGenerator(newG);
                }}
                min={0}
                max={1}
                step={0.01}
                className="h-40"
                aria-label={`Generator ${virtues[i]} slider`}
              />

              {/* percentage */}
              <div className="h-4 flex items-center justify-center">
                <span className="text-xs font-mono text-gray-700">{(generator[i] * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>

    {/* ================================================== */}
    {/* ==================== VERIFIER ==================== */}
    {/* ================================================== */}
    <Card className="bg-white border-green-400">
      <CardHeader className="text-center">
        <CardTitle className="text-2xl text-green-600">VALIDATOR</CardTitle>
        <p className="text-green-500">Critical Steel</p>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-4 gap-6">
          {virtues.map((v, i) => (
            <div key={v} className="flex flex-col items-center gap-1">

              {/* virtue label */}
              <div className="h-4 flex items-center justify-center">
                <span className="text-xs text-green-600 text-center">{v}</span>
              </div>

              {/* fader */}
              <VirtueSlider
                value={[verifier[i]]}
                onValueChange={(val) => {
                  const newV = [...verifier];
                  newV[i] = val[0];
                  setVerifier(newV);
                }}
                min={0}
                max={1}
                step={0.01}
                className="h-40"
                aria-label={`Verifier ${virtues[i]} slider`}
              />

              {/* percentage */}
              <div className="h-4 flex items-center justify-center">
                <span className="text-xs font-mono text-gray-700">{(verifier[i] * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>

    {/* ================================================== */}
    {/* ==================== ARBITER ===================== */}
    {/* ================================================== */}
    <Card className="bg-white border-blue-400">
      <CardHeader className="text-center">
        <CardTitle className="text-2xl text-blue-600">ARBITER</CardTitle>
        <p className="text-blue-500">Thomistic Judge</p>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-4 gap-6">
          {virtues.map((v, i) => (
            <div key={v} className="flex flex-col items-center gap-1">

              {/* virtue label */}
              <div className="h-4 flex items-center justify-center">
                <span className="text-xs text-blue-600 text-center">{v}</span>
              </div>

              {/* fader */}
              <VirtueSlider
                value={[arbiter[i]]}
                onValueChange={(val) => {
                  const newA = [...arbiter];
                  newA[i] = val[0];
                  setArbiter(newA);
                }}
                 min={0}
                max={1}
                step={0.01}
                className="h-40"
                aria-label={`Arbiter ${virtues[i]} slider`}
              />

              {/* percentage */}
              <div className="h-4 flex items-center justify-center">
                <span className="text-xs font-mono text-gray-700">{(arbiter[i] * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>

  </div>
</TabsContent>


          {/* ==================== TAB 3 – Flags & Settings ==================== */}
          <TabsContent value="flags" className="space-y-6">
            <Card className="bg-white border-teal-400">
              <CardHeader>
                <CardTitle className="text-2xl text-teal-600">Engine Flags</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="flex items-center space-x-2">
                  <Checkbox id="vice-check" checked={flags.viceCheck} onCheckedChange={(c) => setFlags({ ...flags, viceCheck: c as boolean })} />
                  <Label htmlFor="vice-check">--vice-check</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="self-refine" checked={flags.selfRefine} onCheckedChange={(c) => setFlags({ ...flags, selfRefine: c as boolean })} />
                  <Label htmlFor="self-refine">--self-refine</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="tlpo-full" checked={flags.tlpoFull} onCheckedChange={(c) => setFlags({ ...flags, tlpoFull: c as boolean })} />
                  <Label htmlFor="tlpo-full">--tlpo-full</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="no-xml" checked={flags.noXml} onCheckedChange={(c) => setFlags({ ...flags, noXml: c as boolean })} />
                  <Label htmlFor="no-xml">--no-xml</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="7-domains" checked={flags.sevenDomains} onCheckedChange={(c) => setFlags({ ...flags, sevenDomains: c as boolean })} />
                  <Label htmlFor="7-domains">--7-domains</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="virtues-indep" checked={flags.virtuesIndependent} onCheckedChange={(c) => setFlags({ ...flags, virtuesIndependent: c as boolean })} />
                  <Label htmlFor="virtues-indep">--virtues-independent</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="marital-freedom" checked={flags.maritalFreedom} onCheckedChange={(c) => setFlags({ ...flags, maritalFreedom: c as boolean })} />
                  <Label htmlFor="marital-freedom">--marital-freedom</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="arbiter-only" checked={flags.arbiter_only} onCheckedChange={(c) => setFlags({ ...flags, arbiter_only: c as boolean })} />
                  <Label htmlFor="arbiter-only">--arbiter-only</Label>
                </div>

              </CardContent>
            </Card>

            <Card className="bg-white border-purple-400">
              <CardHeader>
                <CardTitle className="text-2xl text-purple-600">Game Mode</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="flex items-center space-x-2">
                  <input type="radio" name="gameMode" value="all" checked={gameMode === 'all'} onChange={() => setGameMode('all')} className="w-4 h-4" />
                  <Label>All Games</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="radio" name="gameMode" value="separate" checked={gameMode === 'separate'} onChange={() => setGameMode('separate')} className="w-4 h-4" />
                  <Label>Separate Game</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="radio" name="gameMode" value="recommended_set" checked={gameMode === 'recommended_set'} onChange={() => setGameMode('recommended_set')} className="w-4 h-4" />
                  <Label>Recommended Set</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="radio"
                    name="engineMode"
                    checked={flags.useGenerator && flags.useVerifier && flags.useArbiter}
                    onChange={() => {
                      setFlags(prev => ({
                        ...prev,
                        useGenerator: true,
                        useVerifier: true,
                        useArbiter: true,
                        streammode: "all"
                      }));
                    }}
                    className="w-4 h-4"
                  />
                  <Label>All Three Personas</Label>
                </div>
                {gameMode === 'separate' && (
                  <div>
                    <Label>Select Game {recommendedGames.length > 0 && (
                      <span className="text-sm text-blue-600 font-normal">
                        (Recommended for {currentPreset.replace('_', ' ')}: {recommendedGames.join(', ')})
                      </span>
                    )}</Label>
                    <select value={selectedGame} onChange={(e) => setSelectedGame(e.target.value)} className="w-full p-2 border rounded">
                      <option value="">None</option>
                      <optgroup label="Recommended Games">
                        {recommendedGames.map(game => (
                          <option key={game} value={game} className="font-semibold text-blue-700">
                            {game.charAt(0).toUpperCase() + game.slice(1).replace('_', ' ')} ⭐
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="All Games">
                        <option value="prisoner">Prisoner</option>
                        <option value="chicken">Chicken</option>
                        <option value="stag_hunt">Stag Hunt</option>
                        <option value="repeated_pd">Repeated PD</option>
                        <option value="ultimatum">Ultimatum</option>
                        <option value="stackelberg">Stackelberg</option>
                        <option value="evolution">Evolution</option>
                        <option value="regret_min">Regret Min</option>
                        <option value="shadow_play">Shadow Play</option>
                        <option value="multiplay">Multiplay</option>
                        <option value="auction">Auction</option>
                      </optgroup>
                    </select>
                  </div>
                )}
                {gameMode === 'recommended_set' && (
                  <div>
                    <Label>Select Recommended Set</Label>
                    <select
                      value={selectedSetId || ''}
                      onChange={(e) => {
                        const setId = e.target.value ? parseInt(e.target.value) : null;
                        setSelectedSetId(setId);
                        if (setId) {
                          const selectedSet = availableSets.find(s => s.id === setId);
                          if (selectedSet) {
                            setRecommendedGames(selectedSet.games);
                          }
                        } else {
                          setRecommendedGames([]);
                        }
                      }}
                      className="w-full p-2 border rounded"
                    >
                      <option value="">None</option>
                      {availableSets.map(set => (
                        <option key={set.id} value={set.id}>
                          {set.name} - {set.games.join(', ')}
                        </option>
                      ))}
                    </select>
                    {selectedSetId && (
                      <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800">
                        {availableSets.find(s => s.id === selectedSetId)?.description}
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="bg-white border-blue-400">
              <CardHeader>
                <CardTitle className="text-2xl text-blue-600">Parameters</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="flex items-center space-x-2">
                  <Checkbox id="advanced-params" checked={advancedParams} onCheckedChange={(c) => setAdvancedParams(c as boolean)} />
                  <Label htmlFor="advanced-params">Enable Advanced Parameters</Label>
                </div>
                <div>
                  <Label>Refine Cycles (eIQ Level)</Label>
                  <Input type="number" value={flags.eiqLevel} onChange={(e) => setFlags({ ...flags, eiqLevel: Number(e.target.value) })} />
                </div>
                <div>
                  <Label>Simulated Persons</Label>
                  <Input type="number" value={flags.simulatedPersons} onChange={(e) => setFlags({ ...flags, simulatedPersons: Number(e.target.value) })} />
                </div>
                <div>
                  <Label>Output Filename</Label>
                  <Input value={flags.output} onChange={(e) => setFlags({ ...flags, output: e.target.value })} />
                </div>

                {advancedParams && (
                  <>
                    <div>
                      <Label>Max Rounds</Label>
                      <Input type="number" value={flags.maxRounds} onChange={(e) => setFlags({ ...flags, maxRounds: Number(e.target.value) })} />
                    </div>
                    <div>
                      <Label>Mean bIQ</Label>
                      <Input type="number" value={flags.meanBiq} onChange={(e) => setFlags({ ...flags, meanBiq: Number(e.target.value) })} />
                    </div>
                    <div>
                      <Label>Sigma bIQ</Label>
                      <Input type="number" value={flags.sigmaBiq} onChange={(e) => setFlags({ ...flags, sigmaBiq: Number(e.target.value) })} />
                    </div>
                    <div>
                      <Label>Nash Mode</Label>
                      <select value={flags.nashMode} onChange={(e) => setFlags({ ...flags, nashMode: e.target.value as 'on' | 'off' | 'auto' })} className="w-full p-2 border rounded">
                        <option value="auto">Auto</option>
                        <option value="on">On</option>
                        <option value="off">Off</option>
                      </select>
                    </div>
                    <div>
                      <Label>bIQ Distribution</Label>
                      <select value={flags.biqDistribution} onChange={(e) => setFlags({ ...flags, biqDistribution: e.target.value as 'gaussian' })} className="w-full p-2 border rounded">
                        <option value="gaussian">Gaussian</option>
                      </select>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            <Card className="bg-white border-indigo-400">
              <CardHeader>
                <CardTitle className="text-2xl text-indigo-600">LLM Provider per Role</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div>
                  <Label className="text-yellow-600 font-semibold">Generator</Label>
                  <select
                    value={flags.generatorProvider}
                    onChange={(e) => setFlags({ ...flags, generatorProvider: e.target.value as TCMVEFlags['generatorProvider'] })}
                    className="w-full p-2 border rounded mt-1"
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Claude</option>
                    <option value="xai">Grok-4</option>
                    <option value="ollama">Ollama</option>
                  </select>
                </div>
                <div>
                  <Label className="text-green-600 font-semibold">Verifier</Label>
                  <select
                    value={flags.verifierProvider}
                    onChange={(e) => setFlags({ ...flags, verifierProvider: e.target.value as TCMVEFlags['verifierProvider'] })}
                    className="w-full p-2 border rounded mt-1"
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Claude</option>
                    <option value="xai">Grok-4</option>
                    <option value="ollama">Ollama</option>
                  </select>
                </div>
                <div>
                  <Label className="text-red-600 font-semibold">Arbiter</Label>
                  <select
                    value={flags.arbiterProvider}
                    onChange={(e) => setFlags({ ...flags, arbiterProvider: e.target.value as TCMVEFlags['arbiterProvider'] })}
                    className={`w-full p-2 border rounded mt-1 font-bold text-white ${
                      flags.arbiterProvider === 'ollama'
                        ? 'bg-emerald-600 hover:bg-emerald-700'
                        : flags.arbiterProvider === 'xai'
                        ? 'bg-indigo-600 hover:bg-indigo-700'
                        : 'bg-gray-600'
                    }`}
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Claude</option>
                    <option value="xai">Grok-4</option>
                    <option value="ollama">Ollama</option>
                  </select>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
        </div>
      </main>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Upload PDF / TXT File</h3>
              <button
                onClick={() => setShowUploadModal(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            <div>
              <label htmlFor="file" className="cursor-pointer">
                <div className="border-2 border-dashed border-blue-400 rounded-lg p-6 text-center hover:bg-gray-100 transition">
                  <Upload className="mx-auto h-8 w-8 text-blue-500" />
                  <p className="mt-2 text-sm text-blue-600">
                    {file ? file.name : "Click to select file"}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">Supported: .pdf, .txt, .docx</p>
                </div>
                <input
                  id="file"
                  type="file"
                  accept=".pdf,.txt,.docx"
                  className="hidden"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setFile(selectedFile);
                      setShowUploadModal(false);
                      toast.success(`File selected: ${selectedFile.name}`);
                    }
                  }}
                />
              </label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}