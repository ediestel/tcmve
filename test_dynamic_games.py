#!/usr/bin/env python3
"""
Test Dynamic Game Selection System
ARCHER-1.0 Intelligence Enhancement Framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.game_selector import game_selector

def test_dynamic_game_selection():
    """Test the dynamic game selection system"""

    print("🏹 Testing Dynamic Game Selection System")
    print("=" * 50)

    # Test Case 1: High Justice/Love profile (cooperative focus)
    print("\n📊 Test 1: High Justice/Love Profile (Cooperative Focus)")
    virtue_vectors_cooperative = {
        'generator': [0.8, 0.7, 0.9, 0.6, 0.5, 0.8, 0.4, 0.3],  # High Justice (J), Love (L)
        'verifier': [0.7, 0.8, 0.9, 0.7, 0.6, 0.9, 0.5, 0.4],
        'arbiter': [0.9, 0.8, 0.9, 0.8, 0.7, 0.9, 0.6, 0.5]
    }

    recommendations = game_selector.select_games_dynamic(
        virtue_vectors=virtue_vectors_cooperative,
        query_context="How can we achieve fair cooperation in society?",
        max_games=3,
        execution_mode='sequential'
    )

    print(f"Selected Games: {[r.game_name for r in recommendations]}")
    for rec in recommendations:
        print(f"  {rec.game_name}: {rec.rationale}")

    # Test Case 2: High Prudence/Humility profile (strategic focus)
    print("\n📊 Test 2: High Prudence/Humility Profile (Strategic Focus)")
    virtue_vectors_strategic = {
        'generator': [0.9, 0.9, 0.6, 0.7, 0.8, 0.5, 0.4, 0.3],  # High Ω, P, T
        'verifier': [0.8, 0.9, 0.7, 0.8, 0.9, 0.6, 0.5, 0.4],
        'arbiter': [0.9, 0.9, 0.8, 0.8, 0.8, 0.7, 0.6, 0.5]
    }

    recommendations = game_selector.select_games_dynamic(
        virtue_vectors=virtue_vectors_strategic,
        query_context="What is the optimal strategy for long-term success?",
        max_games=3,
        execution_mode='sequential'
    )

    print(f"Selected Games: {[r.game_name for r in recommendations]}")
    for rec in recommendations:
        print(f"  {rec.game_name}: {rec.rationale}")

    # Test Case 3: Sequential execution plan
    print("\n📊 Test 3: Sequential Execution Plan")
    execution_plan = game_selector.get_sequential_plan(recommendations)
    for step in execution_plan:
        print(f"Step {step['step']}: {step['game']}")
        print(f"  Rationale: {step['rationale']}")
        print(f"  Resource Allocation: {step['resource_allocation']}")
        print(f"  Depends on: {step['depends_on']}")
        print()

    print("✅ Dynamic Game Selection Tests Complete")
    print("\n🎯 Key Benefits:")
    print("- Virtue-aligned game selection")
    print("- Sequential execution prevents potency overload")
    print("- Context-aware recommendations")
    print("- Resource-managed execution")
    print("- Nash equilibrium checkpoints")

if __name__ == "__main__":
    test_dynamic_game_selection()