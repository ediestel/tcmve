// app/api/export/run/[id]/md/route.ts
import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const res = await fetch(`${BACKEND_URL}/export/run/${id}/md`);
  const data = await res.text();
  return new NextResponse(data, {
    headers: {
      'Content-Type': 'text/markdown',
      'Content-Disposition': `attachment; filename="run_${id}.md"`,
    },
  });
}