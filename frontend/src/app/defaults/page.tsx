// src/app/defaults/page.tsx — Virtue Organ moved here for better organization

'use client';

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";

import { toast } from "sonner";
import { api } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import { VirtueSlider } from "@/components/ui/virtue-slider";
import { Plus, Edit, Trash2 } from "lucide-react";
import type { AvailablePresets, Virtues } from "@/lib/types";

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



export default function Defaults() {
  // Virtue vectors for defaults
  const [generator, setGenerator] = useState<number[]>([0.97, 0.8, 0.75, 0.65, 0.85, 0.72, 0.85, 0.89]);
  const [verifier, setVerifier] = useState<number[]>([0.95, 0.9, 0.95, 0.8, 0.9, 0.65, 0.9, 0.95]);
  const [arbiter, setArbiter] = useState<number[]>([0.95, 0.85, 0.8, 0.9, 0.85, 0.85, 0.8, 0.85]);

  // Presets
  const [availablePresets, setAvailablePresets] = useState<AvailablePresets>({});
  const [currentPreset, setCurrentPreset] = useState<string>('');

  // CRUD state
  const [isPresetModalOpen, setIsPresetModalOpen] = useState(false);
  const [editingPreset, setEditingPreset] = useState<string | null>(null);
  const [presetForm, setPresetForm] = useState({
    name: '',
    description: '',
    generator: virtues.reduce((acc, v) => ({ ...acc, [v.split(' ')[0]]: 0.5 }), {} as Virtues),
    verifier: virtues.reduce((acc, v) => ({ ...acc, [v.split(' ')[0]]: 0.5 }), {} as Virtues),
    arbiter: virtues.reduce((acc, v) => ({ ...acc, [v.split(' ')[0]]: 0.5 }), {} as Virtues),
    recommended_games: [] as string[],
    use_case: ''
  });

  useEffect(() => {
    const load = async () => {
      try {
        const [defaults, presets] = await Promise.all([
          api.getDefaults(),
          api.getPresets(),
        ]);
        // Load defaults into sliders
        if (defaults.generator) setGenerator(Object.values(defaults.generator));
        if (defaults.verifier) setVerifier(Object.values(defaults.verifier));
        if (defaults.arbiter) setArbiter(Object.values(defaults.arbiter));
        setAvailablePresets(presets.presets);
      } catch (error) {
        console.error(error);
      }
    };
    load();
  }, []);



  const applyVirtuePreset = async (presetName: string) => {
    try {
      const result = await api.applyPreset(presetName);

      const preset = result.virtue_vectors;
      setGenerator(Object.values(preset.generator) as number[]);
      setVerifier(Object.values(preset.verifier) as number[]);
      setArbiter(Object.values(preset.arbiter) as number[]);
      setCurrentPreset(presetName);

      toast.success(`Applied ${presetName.replace('_', ' ')} preset`);
    } catch (error) {
      toast.error(`Failed to apply ${presetName} preset`);
      console.error(error);
    }
  };

  const openPresetModal = (presetName?: string) => {
    if (presetName) {
      // Edit mode
      setEditingPreset(presetName);
      api.getPreset(presetName).then(preset => {
        setPresetForm({
          name: preset.name,
          description: preset.description,
          generator: preset.virtue_vectors.generator,
          verifier: preset.virtue_vectors.verifier,
          arbiter: preset.virtue_vectors.arbiter,
          recommended_games: preset.recommended_games || [],
          use_case: preset.use_case || ''
        });
      });
    } else {
      // Create mode
      setEditingPreset(null);
      setPresetForm({
        name: '',
        description: '',
        generator: virtues.reduce((acc, v) => ({ ...acc, [v.split(' ')[0]]: 0.5 }), {} as Virtues),
        verifier: virtues.reduce((acc, v) => ({ ...acc, [v.split(' ')[0]]: 0.5 }), {} as Virtues),
        arbiter: virtues.reduce((acc, v) => ({ ...acc, [v.split(' ')[0]]: 0.5 }), {} as Virtues),
        recommended_games: [],
        use_case: ''
      });
    }
    setIsPresetModalOpen(true);
  };

  const savePreset = async () => {
    try {
      const virtueVectors = {
        generator: presetForm.generator,
        verifier: presetForm.verifier,
        arbiter: presetForm.arbiter
      };

      if (editingPreset) {
        await api.updatePreset(editingPreset, {
          description: presetForm.description,
          virtue_vectors: virtueVectors,
          recommended_games: presetForm.recommended_games,
          use_case: presetForm.use_case
        });
        toast.success(`Updated preset ${presetForm.name}`);
      } else {
        await api.createPreset({
          name: presetForm.name,
          description: presetForm.description,
          virtue_vectors: virtueVectors,
          recommended_games: presetForm.recommended_games,
          use_case: presetForm.use_case
        });
        toast.success(`Created preset ${presetForm.name}`);
      }

      // Refresh presets
      const updatedPresets = await api.getPresets();
      setAvailablePresets(updatedPresets.presets);
      setIsPresetModalOpen(false);
    } catch (error) {
      toast.error(`Failed to save preset`);
      console.error(error);
    }
  };

  const deletePreset = async (presetName: string) => {
    if (!confirm(`Delete preset "${presetName}"?`)) return;

    try {
      await api.deletePreset(presetName);
      toast.success(`Deleted preset ${presetName}`);

      // Refresh presets
      const updatedPresets = await api.getPresets();
      setAvailablePresets(updatedPresets.presets);
    } catch (error) {
      toast.error(`Failed to delete preset`);
      console.error(error);
    }
  };

  const saveDefaults = async () => {
    try {
      const defaults = {
        generator: virtues.reduce((acc, v, i) => ({ ...acc, [v.split(' ')[0]]: generator[i] }), {} as Virtues),
        verifier: virtues.reduce((acc, v, i) => ({ ...acc, [v.split(' ')[0]]: verifier[i] }), {} as Virtues),
        arbiter: virtues.reduce((acc, v, i) => ({ ...acc, [v.split(' ')[0]]: arbiter[i] }), {} as Virtues),
      };
      await api.saveDefaults(defaults);
      toast.success("Defaults saved");
    } catch (error) {
      toast.error("Failed to save defaults");
      console.error(error);
    }
  };

  const calculateV = (values: number[]) => {
    const product = values.reduce((acc, v) => acc * v, 1);
    return (product * 1000).toFixed(3);
  };

  const V = Math.min(
    Number(calculateV(generator)),
    Number(calculateV(verifier)),
    Number(calculateV(arbiter))
  ).toFixed(3);

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 via-gray-100 to-gray-50 text-gray-900 flex">
      <Sidebar />
      <main className="flex-1 p-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-4xl font-bold text-blue-600 mb-8">Virtue Organ & Defaults</h1>
          <p className="text-center text-sm mb-8 text-blue-600">
            @ECKHART_DIESTEL — V = {V}
          </p>

          <div className="mb-6">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-xl font-semibold">Domain-Specific Virtue Presets</h2>
              <Button onClick={() => openPresetModal()} variant="outline" size="sm">
                <Plus className="w-4 h-4 mr-2" />
                Add Preset
              </Button>
            </div>
            <p className="text-sm text-gray-600 mb-4">
              Click a preset to apply optimized virtue configurations for specific ethical domains.
              {currentPreset && <span className="ml-2 font-medium text-blue-600">Current: {currentPreset.replace('_', ' ')}</span>}
            </p>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
              {Object.entries(availablePresets).map(([presetName, description]) => (
                <div key={presetName} className="relative group">
                  <Button
                    onClick={() => applyVirtuePreset(presetName)}
                    variant={currentPreset === presetName ? "default" : "outline"}
                    className={`text-xs p-2 h-auto w-full ${
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
                  <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                    <Button
                      onClick={(e) => { e.stopPropagation(); openPresetModal(presetName); }}
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 bg-white hover:bg-gray-100"
                    >
                      <Edit className="w-3 h-3" />
                    </Button>
                    <Button
                      onClick={(e) => { e.stopPropagation(); deletePreset(presetName); }}
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 bg-white hover:bg-red-100 text-red-600"
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            {Object.keys(availablePresets).length === 0 && (
              <p className="text-sm text-gray-500 italic">No presets available. Create your first preset above.</p>
            )}

          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            {/* Generator */}
            <Card className="bg-white border-yellow-400">
              <CardHeader className="text-center">
                <CardTitle className="text-2xl text-yellow-600">GENERATOR</CardTitle>
                <p className="text-yellow-500">Creative Fire</p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-6">
                  {virtues.map((v, i) => (
                    <div key={v} className="flex flex-col items-center gap-1">
                      <div className="h-4 flex items-center justify-center">
                        <span className="text-xs text-yellow-600 text-center">{v}</span>
                      </div>
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
                      <div className="h-4 flex items-center justify-center">
                        <span className="text-xs font-mono text-gray-700">{(generator[i] * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Verifier */}
            <Card className="bg-white border-green-400">
              <CardHeader className="text-center">
                <CardTitle className="text-2xl text-green-600">VALIDATOR</CardTitle>
                <p className="text-green-500">Critical Steel</p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-6">
                  {virtues.map((v, i) => (
                    <div key={v} className="flex flex-col items-center gap-1">
                      <div className="h-4 flex items-center justify-center">
                        <span className="text-xs text-green-600 text-center">{v}</span>
                      </div>
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
                      <div className="h-4 flex items-center justify-center">
                        <span className="text-xs font-mono text-gray-700">{(verifier[i] * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Arbiter */}
            <Card className="bg-white border-blue-400">
              <CardHeader className="text-center">
                <CardTitle className="text-2xl text-blue-600">ARBITER</CardTitle>
                <p className="text-blue-500">Thomistic Judge</p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-6">
                  {virtues.map((v, i) => (
                    <div key={v} className="flex flex-col items-center gap-1">
                      <div className="h-4 flex items-center justify-center">
                        <span className="text-xs text-blue-600 text-center">{v}</span>
                      </div>
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
                      <div className="h-4 flex items-center justify-center">
                        <span className="text-xs font-mono text-gray-700">{(arbiter[i] * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="text-center">
            <Button onClick={saveDefaults} className="bg-blue-600 hover:bg-blue-700 text-white">
              Save Defaults
            </Button>
          </div>
        </div>
      </main>

      {/* Preset CRUD Modal */}
      {isPresetModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">{editingPreset ? 'Edit Preset' : 'Create New Preset'}</h3>
              <button
                onClick={() => setIsPresetModalOpen(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="preset-name">Name</Label>
                <Input
                  id="preset-name"
                  value={presetForm.name}
                  onChange={(e) => setPresetForm({ ...presetForm, name: e.target.value })}
                  placeholder="e.g., healthcare_ethics"
                  disabled={!!editingPreset}
                />
              </div>
              <div>
                <Label htmlFor="preset-description">Description</Label>
                <Input
                  id="preset-description"
                  value={presetForm.description}
                  onChange={(e) => setPresetForm({ ...presetForm, description: e.target.value })}
                  placeholder="Brief description of the preset"
                />
              </div>
            </div>

            <div>
              <Label htmlFor="preset-use-case">Use Case</Label>
              <Textarea
                id="preset-use-case"
                value={presetForm.use_case}
                onChange={(e) => setPresetForm({ ...presetForm, use_case: e.target.value })}
                placeholder="When to use this preset"
                rows={2}
              />
            </div>

            <div>
              <Label>Recommended Games</Label>
              <div className="flex flex-wrap gap-2 mt-1">
                {['prisoner', 'chicken', 'stag_hunt', 'repeated_pd', 'ultimatum', 'stackelberg', 'evolution', 'regret_min', 'shadow_play', 'multiplay', 'auction'].map(game => (
                  <label key={game} className="flex items-center space-x-1">
                    <input
                      type="checkbox"
                      checked={presetForm.recommended_games.includes(game)}
                      onChange={(e) => {
                        const games = e.target.checked
                          ? [...presetForm.recommended_games, game]
                          : presetForm.recommended_games.filter(g => g !== game);
                        setPresetForm({ ...presetForm, recommended_games: games });
                      }}
                      className="rounded"
                    />
                    <span className="text-sm capitalize">{game.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Virtue Sliders for Preset */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader className="text-center">
                  <CardTitle className="text-lg text-yellow-600">Generator Virtues</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    {virtues.map((v, i) => (
                      <div key={v} className="flex flex-col items-center gap-1">
                        <span className="text-xs text-center">{v}</span>
                        <VirtueSlider
                          value={[Object.values(presetForm.generator)[i] || 0.5]}
                          onValueChange={(val) => {
                            const newGen = { ...presetForm.generator };
                            newGen[virtues[i].split(' ')[0] as keyof Virtues] = val[0];
                            setPresetForm({ ...presetForm, generator: newGen });
                          }}
                          min={0}
                          max={1}
                          step={0.01}
                          className="h-20"
                        />
                        <span className="text-xs font-mono">{(Object.values(presetForm.generator)[i] || 0.5).toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="text-center">
                  <CardTitle className="text-lg text-green-600">Verifier Virtues</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    {virtues.map((v, i) => (
                      <div key={v} className="flex flex-col items-center gap-1">
                        <span className="text-xs text-center">{v}</span>
                        <VirtueSlider
                          value={[Object.values(presetForm.verifier)[i] || 0.5]}
                          onValueChange={(val) => {
                            const newVer = { ...presetForm.verifier };
                            newVer[virtues[i].split(' ')[0] as keyof Virtues] = val[0];
                            setPresetForm({ ...presetForm, verifier: newVer });
                          }}
                          min={0}
                          max={1}
                          step={0.01}
                          className="h-20"
                        />
                        <span className="text-xs font-mono">{(Object.values(presetForm.verifier)[i] || 0.5).toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="text-center">
                  <CardTitle className="text-lg text-blue-600">Arbiter Virtues</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    {virtues.map((v, i) => (
                      <div key={v} className="flex flex-col items-center gap-1">
                        <span className="text-xs text-center">{v}</span>
                        <VirtueSlider
                          value={[Object.values(presetForm.arbiter)[i] || 0.5]}
                          onValueChange={(val) => {
                            const newArb = { ...presetForm.arbiter };
                            newArb[virtues[i].split(' ')[0] as keyof Virtues] = val[0];
                            setPresetForm({ ...presetForm, arbiter: newArb });
                          }}
                          min={0}
                          max={1}
                          step={0.01}
                          className="h-20"
                        />
                        <span className="text-xs font-mono">{(Object.values(presetForm.arbiter)[i] || 0.5).toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setIsPresetModalOpen(false)}>
                Cancel
              </Button>
              <Button onClick={savePreset} disabled={!presetForm.name || !presetForm.description}>
                {editingPreset ? 'Update Preset' : 'Create Preset'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    )}
    </div>
  );
}