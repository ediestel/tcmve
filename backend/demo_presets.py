#!/usr/bin/env python3
"""
Demonstration of Virtue Presets for Domain-Specific TCMVE Analysis

This script shows how to programmatically determine and apply preset profiles
for different ethical domains.
"""

from tcmve import TCMVE
from virtue_presets import get_preset, list_presets, get_virtue_vectors_for_preset

def demonstrate_presets():
    """Demonstrate how to use virtue presets programmatically."""

    print("🔍 TCMVE Virtue Presets Demonstration")
    print("=" * 50)

    # 1. List all available presets
    print("\n📋 Available Virtue Presets:")
    presets = list_presets()
    for name, description in presets.items():
        print(f"  • {name}: {description}")

    # 2. Get detailed information about a specific preset
    print("\n🏥 Healthcare Ethics Preset Details:")
    healthcare_preset = get_preset("healthcare_ethics")
    print(f"Name: {healthcare_preset['name']}")
    print(f"Description: {healthcare_preset['description']}")
    print(f"Recommended Games: {healthcare_preset['recommended_games']}")
    print(f"Use Case: {healthcare_preset['use_case']}")

    print("\n💰 Virtue Vectors for Healthcare Ethics:")
    virtue_vectors = get_virtue_vectors_for_preset("healthcare_ethics")
    for role, virtues in virtue_vectors.items():
        print(f"  {role.title()}: {virtues}")

    # 3. Demonstrate programmatic application
    print("\n🚀 Applying Preset to TCMVE Engine:")

    # Initialize TCMVE engine
    engine = TCMVE(max_rounds=3, cache_enabled=True)

    print("Before preset application:")
    print(f"  Generator Prudence: {engine.virtue_vectors['generator']['P']}")
    print(f"  Arbiter Justice: {engine.virtue_vectors['arbiter']['J']}")

    # Apply healthcare ethics preset
    engine.apply_virtue_preset("healthcare_ethics")

    print("\nAfter applying 'healthcare_ethics' preset:")
    print(f"  Generator Prudence: {engine.virtue_vectors['generator']['P']}")
    print(f"  Arbiter Justice: {engine.virtue_vectors['arbiter']['J']}")

    # 4. Show domain-specific reasoning
    print("\n🧠 Domain-Specific Ethical Reasoning:")

    domains = {
        "autonomous_vehicles": "Should a self-driving car swerve to avoid pedestrians?",
        "financial_risk": "Should a bank invest in high-risk assets during recession?",
        "legal_justice": "Should a judge consider defendant's socioeconomic status?",
        "environmental_policy": "Should a government prioritize economic growth over emissions?",
        "academic_integrity": "Should a professor accept late work for struggling students?"
    }

    for domain, question in domains.items():
        preset = get_preset(domain)
        print(f"\n{domain.replace('_', ' ').title()}:")
        print(f"  Question: {question}")
        print(f"  Key Virtue Focus: {preset['description'].split(',')[0]}")
        print(f"  Recommended Analysis: {', '.join(preset['recommended_games'])}")

def automated_analysis_demo():
    """Demonstrate automated analysis with different presets."""

    print("\n🤖 Automated Analysis with Different Presets")
    print("=" * 50)

    test_query = "Should experimental treatment be given to terminally ill patients?"

    presets_to_test = ["healthcare_ethics", "legal_justice", "financial_risk"]

    for preset_name in presets_to_test:
        print(f"\n🔬 Analyzing with {preset_name.replace('_', ' ').title()} preset:")

        # Get preset configuration
        virtue_config = get_virtue_vectors_for_preset(preset_name)

        # In a real implementation, you would:
        # 1. Initialize TCMVE with preset virtues
        # 2. Run analysis
        # 3. Cache results for future use

        print(f"  Virtue Configuration: Generator P={virtue_config['generator']['P']}, Arbiter J={virtue_config['arbiter']['J']}")
        print("  Analysis would focus on:")
        preset = get_preset(preset_name)
        print(f"    {preset['description']}")
        print("  Result would be cached for future identical queries")

if __name__ == "__main__":
    demonstrate_presets()
    automated_analysis_demo()

    print("\n✅ Preset demonstration complete!")
    print("\n💡 To use presets in your application:")
    print("   from virtue_presets import get_virtue_vectors_for_preset")
    print("   from tcmve import TCMVE")
    print("   ")
    print("   engine = TCMVE()")
    print("   engine.apply_virtue_preset('healthcare_ethics')")
    print("   result = engine.run('your ethical dilemma query')")