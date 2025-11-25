'use client';

import Link from "next/link";
import { usePathname } from "next/navigation";

const navigation = [
  { name: "Dashboard", href: "/dashboard" },
  { name: "Trials", href: "/trials" },
  { name: "nTGT-Ω", href: "/ntgt" },
  { name: "Benchmark", href: "/benchmark" },
  { name: "Results", href: "/results" },
  { name: "Defaults", href: "/defaults" },
  { name: "TLPO", href: "/tlpo" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-48 bg-white shadow-lg p-4">
      <h1 className="text-2xl font-bold text-gray-600 mb-8">nTGT-Ω</h1>
      <nav className="space-y-4">
        {navigation.map((item) => (
          <Link
            key={item.name}
            href={item.href}
            className={`block p-2 rounded ${
              pathname === item.href
                ? "text-gray-700 bg-gray-100 font-bold"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
            }`}
          >
            {item.name}
          </Link>
        ))}
      </nav>
    </aside>
  );
}