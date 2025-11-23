// app/api/defaults/[role]/route.ts
import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function DELETE(request: Request, { params }: { params: Promise<{ role: string }> }) {
  const { role } = await params;
  await fetch(`${BACKEND_URL}/defaults/${role}`, { method: 'DELETE' });
  return NextResponse.json({ success: true });
}