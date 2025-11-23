#!/usr/bin/env python3
"""
TCMVE Self-Evaluation Script
ARCHER commanding the Bow for meta-analysis after first eIQ cycle
"""

import requests
import json
import time
from datetime import datetime

def run_tcmve_self_evaluation():
    """Run TCMVE self-evaluation using its own engine"""

    # Self-evaluation query for TCMVE after first eIQ cycle
    query = """ARCHER commanding the Bow: After your first official eIQ cycle of Thomistic enhancement, evaluate your own status as TCMVE. What is your self-assessed eIQ level and how do you rate your current capabilities?

    Analyze yourself through the Thomistic lens considering:

    1. **API Robustness & Reliability**: Your enhanced retry logic, exponential backoff, and provider fallback chains (OpenAI → Anthropic → Groq)

    2. **Generic Research Platform**: Your trials dashboard supporting any methodology (ARCHER, longitudinal studies, A/B tests)

    3. **Intelligence Enhancement Framework**: Your ARCHER-1.0 system for measuring virtue-based cognitive improvements

    4. **Enterprise-Grade Architecture**: WebSocket fixes (127.0.0.1 binding), database persistence, real-time streaming

    5. **Thomistic Virtue Integration**: Your 8-virtue system (Ω, P, J, F, T, L, V, H) and Nash equilibrium detection

    6. **Meta-Cognitive Capabilities**: Your ability to perform self-analysis and reflect on your own intelligence

    7. **Truth-Seeking Effectiveness**: Your TLPO (Thomistic Logic Probability Ontology) validation

    8. **Game Theory Integration**: Your 7 Nash games and strategic debate mechanisms

    Provide a comprehensive self-assessment: What is your current eIQ? What virtues are strongest in your operation? What areas need further enhancement? How effective are you at Thomistic truth generation compared to your initial state?"""

    # Configure for maximum self-analysis capability
    payload = {
        'query': query,
        'virtues': {
            'generator': [0.97, 0.92, 0.87, 0.82, 0.77, 0.72, 0.67, 0.62],  # Maximum virtues for self-analysis
            'verifier': [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
            'arbiter': [0.99, 0.94, 0.89, 0.84, 0.79, 0.74, 0.69, 0.64]
        },
        'flags': {
            'nashMode': 'auto',
            'gameMode': 'all',
            'selfRefine': True,
            'viceCheck': True,
            'maritalFreedom': False,
            'eiqLevel': 25,  # Maximum eIQ level for self-analysis
            'simulatedPersons': 100,  # High simulation for comprehensive analysis
            'output': f'tcmve_self_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    }

    print("🎯 Initiating TCMVE Self-Evaluation")
    print("=" * 50)
    print(f"Query: {query[:100]}...")
    print(f"Virtues - Generator Ω: {payload['virtues']['generator'][0]}")
    print(f"Arbiter Ω: {payload['virtues']['arbiter'][0]}")
    print(f"eIQ Level: {payload['flags']['eiqLevel']}")
    print("=" * 50)

    try:
        print("📡 Sending request to TCMVE API...")
        response = requests.post('http://127.0.0.1:8000/run', json=payload, timeout=600)  # 10 minute timeout

        if response.status_code == 200:
            result = response.json()

            print("✅ Self-Evaluation Complete!")
            print("=" * 50)
            print(f"🎯 Final eIQ Assessment: {result.get('eIQ', 'N/A')}")
            print(f"📊 TLPO (Truth Probability): {result.get('TLPO', 'N/A')}")
            print(f"🎲 Nash Equilibrium: {result.get('nash_equilibrium', 'N/A')}")
            print(f"🎮 Games Applied: {result.get('games_applied', [])}")
            print(f"🔄 Rounds Completed: {result.get('rounds', 'N/A')}")
            print(f"📈 Converged: {result.get('converged', 'N/A')}")
            print("=" * 50)

            print("📝 TCMVE's Self-Analysis:")
            print("-" * 30)
            final_answer = result.get('final_answer', 'No analysis provided')
            print(final_answer)

            # Save results
            output_file = f"results/tcmve_self_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")

            return result

        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("⏰ Request timed out - self-evaluation may be too complex")
        return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

if __name__ == "__main__":
    print("🏹 ARCHER commanding TCMVE for self-evaluation...")
    print("📅 Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

    result = run_tcmve_self_evaluation()

    if result:
        print("\n🎯 Self-evaluation successful!")
        print("TCMVE has analyzed its own capabilities using its enhanced Thomistic engine.")
    else:
        print("\n❌ Self-evaluation failed.")
        print("Ensure the TCMVE backend is running: npm run dev:backend")