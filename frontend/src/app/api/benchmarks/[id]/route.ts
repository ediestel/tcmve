// app/api/benchmarks/[id]/route.ts
import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function DELETE(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  await fetch(`${BACKEND_URL}/benchmarks/${id}`, { method: 'DELETE' });
  return NextResponse.json({ success: true });
}