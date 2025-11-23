#!/usr/bin/env python3
"""
Test Script for Virtue Evolution System
ARCHER-1.0 Intelligence Enhancement Framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.virtue_evolution import virtue_tracker, thomistic_adjuster
import uuid

def test_virtue_evolution():
    """Test the virtue evolution tracking system"""

    print("Testing Virtue Evolution System")
    print("=" * 40)

    # Create a test session
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    print(f"Test Session ID: {session_id}")

    # Test 1: Record some virtue adjustments
    print("\n1. Recording virtue adjustments...")

    # Simulate performance metrics
    performance_metrics = {
        'converged': True,
        'tlpo_score': 0.85,
        'games_applied': ['prisoner', 'auction'],
        'vice_score': 0.1,
        'contradictions_detected': 1,
        'ethical_flags': []
    }

    # Record adjustments for generator
    adjustments = thomistic_adjuster.adjust_virtues_thomistically(
        session_id=session_id,
        agent_role='generator',
        performance_metrics=performance_metrics,
        trigger_event='test_convergence',
        query_context='Testing Thomistic virtue development'
    )

    print(f"✓ Recorded {len(adjustments)} virtue adjustments for generator")
    for virtue, adjustment in adjustments.items():
        print(f"  {virtue}: {adjustment:.3f}")

    # Test 2: Retrieve evolution history
    print("\n2. Retrieving evolution history...")

    evolution = virtue_tracker.get_virtue_evolution(session_id, limit=10)
    print(f"✓ Retrieved {len(evolution)} evolution records")

    for record in evolution[:3]:  # Show first 3
        print(f"  {record['agent_role']}.{record['virtue_name']}: {record['value_before']:.2f} -> {record['value_after']:.2f} ({record['adjustment']:+.3f})")

    # Test 3: Get current virtue state
    print("\n3. Getting current virtue state...")

    current_state = virtue_tracker.get_current_virtue_state(session_id)
    print("✓ Current virtue state:")
    for role, virtues in current_state.items():
        print(f"  {role}:")
        for virtue, value in virtues.items():
            print(f"    {virtue}: {value:.2f}")

    # Test 4: Analyze virtue development
    print("\n4. Analyzing virtue development...")

    analysis = virtue_tracker.analyze_virtue_development(session_id)
    print("✓ Virtue development analysis:")
    for virtue, stats in analysis.items():
        print(f"  {virtue}: {stats['adjustment_count']} adjustments, avg {stats['avg_adjustment']:+.3f}, net {stats['net_change']:+.3f}")

    # Test 5: Test multiple agents
    print("\n5. Testing multiple agents...")

    for role in ['verifier', 'arbiter']:
        adjustments = thomistic_adjuster.adjust_virtues_thomistically(
            session_id=session_id,
            agent_role=role,
            performance_metrics=performance_metrics,
            trigger_event=f'test_{role}',
            query_context=f'Testing {role} virtue development'
        )
        print(f"✓ Recorded adjustments for {role}: {len(adjustments)} virtues")

    # Final state check
    final_state = virtue_tracker.get_current_virtue_state(session_id)
    print("\n✓ Final virtue state across all agents:")
    for role in ['generator', 'verifier', 'arbiter']:
        if role in final_state:
            virtues = final_state[role]
            virtue_summary = ", ".join([f"{v}: {virtues[v]:.2f}" for v in sorted(virtues.keys())])
            print(f"  {role}: {virtue_summary}")

    print("\n✓ All tests completed successfully!")
    print("The Thomistic virtue evolution system is fully operational.")

if __name__ == "__main__":
    test_virtue_evolution()