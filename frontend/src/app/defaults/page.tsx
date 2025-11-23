// src/app/defaults/page.tsx — SMALL, CLEAN, SHOWS ALL 3 ROLES SIDE-BY-SIDE

'use client';

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { api } from "@/lib/api";
import Sidebar from "@/components/Sidebar";

interface Virtues {
  Ω: number; P: number; J: number; F: number;
  T: number; L: number; V: number; H: number;
}

export default function Defaults() {
  const [generator, setGenerator] = useState<Virtues | null>(null);
  const [verifier, setVerifier] = useState<Virtues | null>(null);
  const [arbiter, setArbiter] = useState<Virtues | null>(null);
  const [editing, setEditing] = useState<'generator' | 'verifier' | 'arbiter' | null>(null);

  useEffect(() => {
    fetchDefaults();
  }, []);

  const fetchDefaults = async () => {
    try {
      const data = await api.getDefaults();
      setGenerator((data.generator as Virtues) || { Ω:0.97, P:0.8, J:0.75, F:0.65, T:0.85, L:0.72, V:0.85, H:0.89 });
      setVerifier((data.verifier as Virtues) || { Ω:0.95, P:0.9, J:0.95, F:0.8, T:0.9, L:0.65, V:0.9, H:0.95 });
      setArbiter((data.arbiter as Virtues) || { Ω:0.95, P:0.85, J:0.8, F:0.9, T:0.85, L:0.85, V:0.8, H:0.85 });
    } catch {
      toast.error("Failed to load defaults");
    }
  };

  const save = async (role: string, virtues: Virtues) => {
    try {
      // Fetch current defaults and update the specific role
      const currentDefaults = await api.getDefaults();
      const updatedDefaults = {
        ...currentDefaults,
        [role]: virtues
      };
      await api.saveDefaults(updatedDefaults);
      toast.success(`${role} saved`);
      setEditing(null);
      // Refresh the local state
      fetchDefaults();
    } catch {
      toast.error("Save failed");
    }
  };

  const VirtueRow = ({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) => (
    <div className="flex items-center gap-3">
      <Label className="w-8 text-right">{label}</Label>
      <Input
        type="number"
        step="0.01"
        min="0"
        max="1"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-20 font-mono"
      />
      <span className="text-sm font-mono w-12">{value.toFixed(3)}</span>
    </div>
  );

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 via-gray-100 to-gray-50 text-gray-900 flex">
      <Sidebar />

      <main className="flex-1 p-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-4xl font-bold text-blue-600 mb-8">Default Virtue Sets</h1>

          <div className="grid md:grid-cols-3 gap-8">
            {(['generator', 'verifier', 'arbiter'] as const).map(role => {
              const virtues = role === 'generator' ? generator : role === 'verifier' ? verifier : arbiter;
              const setVirtues = role === 'generator' ? setGenerator : role === 'verifier' ? setVerifier : setArbiter;

              return (
                <Card key={role} className="bg-white border-blue-300 shadow-lg">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-2xl capitalize text-blue-600 flex justify-between items-center">
                      {role}
                      <Button size="sm" onClick={() => setEditing(role)}>Edit</Button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {virtues && Object.entries(virtues).map(([k, v]) => (
                      <VirtueRow
                        key={k}
                        label={k}
                        value={v}
                        onChange={(nv) => setVirtues?.(prev => ({ ...prev!, [k]: nv }))}
                      />
                    ))}
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Edit Modal */}
          {editing && (
            <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
              <Card className="w-full max-w-lg bg-white">
                <CardHeader>
                  <CardTitle className="text-2xl text-blue-600 capitalize">Edit {editing}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {(editing === 'generator' ? generator : editing === 'verifier' ? verifier : arbiter) && 
                    Object.entries(editing === 'generator' ? generator! : editing === 'verifier' ? verifier! : arbiter!).map(([k, v]) => (
                      <VirtueRow
                        key={k}
                        label={k}
                        value={v}
                        onChange={(nv) => {
                          if (editing === 'generator') setGenerator(prev => ({ ...prev!, [k]: nv }));
                          if (editing === 'verifier') setVerifier(prev => ({ ...prev!, [k]: nv }));
                          if (editing === 'arbiter') setArbiter(prev => ({ ...prev!, [k]: nv }));
                        }}
                      />
                    ))
                  }
                  <div className="flex justify-end gap-3 pt-4">
                    <Button onClick={() => {
                      const virtues = editing === 'generator' ? generator : editing === 'verifier' ? verifier : arbiter;
                      if (virtues) save(editing, virtues);
                    }} className="bg-blue-600 hover:bg-blue-700">
                      Save
                    </Button>
                    <Button onClick={() => setEditing(null)} variant="outline">Cancel</Button>
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