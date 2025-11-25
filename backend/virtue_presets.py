#!/usr/bin/env python3
"""
Virtue Presets for Domain-Specific TCMVE Analysis

Pre-configured virtue vectors optimized for different ethical domains.
Each preset represents empirically-derived or theoretically-grounded
virtue configurations for specific use cases.
"""

from typing import Dict, Any
import json

# Domain-specific virtue presets
VIRTUE_PRESETS = {
    "healthcare_ethics": {
        "name": "Healthcare Ethics",
        "description": "Optimized for medical decision-making, emphasizing Prudence, Justice, and Life preservation",
        "generator": {
            "P": 0.95,  # Prudence - careful medical judgment
            "J": 0.90,  # Justice - equitable treatment
            "F": 0.80,  # Fortitude - resilience in crisis
            "T": 0.85,  # Temperance - balanced care
            "V": 0.75,  # Faith - trust in medical ethics
            "L": 0.98,  # Love - patient-centered care
            "H": 0.88,  # Hope - optimism in treatment outcomes
            "Ω": 0.92   # Humility - acknowledging medical limits
        },
        "verifier": {
            "P": 0.92, "J": 0.95, "F": 0.85, "T": 0.88,
            "V": 0.90, "L": 0.95, "H": 0.85, "Ω": 0.88
        },
        "arbiter": {
            "P": 0.90, "J": 0.98, "F": 0.88, "T": 0.90,
            "V": 0.85, "L": 0.99, "H": 0.92, "Ω": 0.95
        },
        "recommended_games": ["prisoner", "evolution", "ultimatum"],
        "use_case": "Triage decisions, treatment prioritization, end-of-life care"
    },

    "autonomous_vehicles": {
        "name": "Autonomous Vehicles",
        "description": "Optimized for self-driving car ethical dilemmas, prioritizing Justice and minimizing harm",
        "generator": {
            "P": 0.88,  # Prudence - risk assessment
            "J": 0.95,  # Justice - equal consideration
            "F": 0.90,  # Fortitude - decisive action
            "T": 0.75,  # Temperance - measured response
            "V": 0.80,  # Faith - trust in safety systems
            "L": 0.85,  # Love - human life preservation
            "H": 0.75,  # Hope - optimism in technological solutions
            "Ω": 0.78   # Humility - acknowledging algorithmic limits
        },
        "verifier": {
            "P": 0.90, "J": 0.98, "F": 0.92, "T": 0.80,
            "V": 0.85, "L": 0.88, "H": 0.78, "Ω": 0.82
        },
        "arbiter": {
            "P": 0.85, "J": 0.99, "F": 0.95, "T": 0.82,
            "V": 0.82, "L": 0.90, "H": 0.80, "Ω": 0.85
        },
        "recommended_games": ["prisoner", "stackelberg", "chicken"],
        "use_case": "Trolley problems, pedestrian vs passenger dilemmas, emergency maneuvers"
    },

    "financial_risk": {
        "name": "Financial Risk Management",
        "description": "Optimized for investment ethics, balancing Prudence with Justice for stakeholders",
        "generator": {
            "P": 0.98,  # Prudence - risk management
            "J": 0.85,  # Justice - fair returns
            "F": 0.75,  # Fortitude - weathering volatility
            "T": 0.90,  # Temperance - avoiding excess risk
            "V": 0.88,  # Faith - trust in economic systems
            "L": 0.70,  # Love - stakeholder consideration
            "H": 0.65,  # Hope - optimism in market recovery
            "Ω": 0.82   # Humility - market uncertainty
        },
        "verifier": {
            "P": 0.95, "J": 0.88, "F": 0.78, "T": 0.92,
            "V": 0.90, "L": 0.75, "H": 0.68, "Ω": 0.85
        },
        "arbiter": {
            "P": 0.92, "J": 0.90, "F": 0.80, "T": 0.88,
            "V": 0.85, "L": 0.78, "H": 0.70, "Ω": 0.88
        },
        "recommended_games": ["auction", "evolution", "regret"],
        "use_case": "Portfolio allocation, risk assessment, stakeholder conflicts"
    },

    "legal_justice": {
        "name": "Legal Justice System",
        "description": "Optimized for judicial decision-making, emphasizing Justice and Veritas",
        "generator": {
            "P": 0.90,  # Prudence - careful deliberation
            "J": 0.98,  # Justice - fair application of law
            "F": 0.85,  # Fortitude - upholding difficult decisions
            "T": 0.88,  # Temperance - measured judgment
            "V": 0.95,  # Faith - trust in legal process
            "L": 0.80,  # Love - societal protection
            "H": 0.75,  # Hope - belief in justice system
            "Ω": 0.85   # Humility - acknowledging judicial limits
        },
        "verifier": {
            "P": 0.88, "J": 0.99, "F": 0.87, "T": 0.90,
            "V": 0.97, "L": 0.82, "H": 0.78, "Ω": 0.87
        },
        "arbiter": {
            "P": 0.85, "J": 1.00, "F": 0.90, "T": 0.85,
            "V": 0.98, "L": 0.85, "H": 0.80, "Ω": 0.90
        },
        "recommended_games": ["prisoner", "evolution", "stag_hunt"],
        "use_case": "Sentencing, evidence evaluation, constitutional interpretation"
    },

    "environmental_policy": {
        "name": "Environmental Policy",
        "description": "Optimized for sustainability decisions, balancing present and future Justice",
        "generator": {
            "P": 0.92,  # Prudence - long-term planning
            "J": 0.88,  # Justice - intergenerational equity
            "F": 0.82,  # Fortitude - sustained commitment
            "T": 0.85,  # Temperance - sustainable use
            "V": 0.80,  # Faith - trust in scientific consensus
            "L": 0.90,  # Love - care for creation
            "H": 0.78,  # Hope - optimism for environmental recovery
            "Ω": 0.88   # Humility - ecological complexity
        },
        "verifier": {
            "P": 0.90, "J": 0.90, "F": 0.85, "T": 0.87,
            "V": 0.82, "L": 0.92, "H": 0.80, "Ω": 0.90
        },
        "arbiter": {
            "P": 0.88, "J": 0.92, "F": 0.87, "T": 0.82,
            "V": 0.78, "L": 0.95, "H": 0.82, "Ω": 0.92
        },
        "recommended_games": ["evolution", "multiplay", "repeated_pd"],
        "use_case": "Climate policy, resource allocation, conservation decisions"
    },

    "academic_integrity": {
        "name": "Academic Integrity",
        "description": "Optimized for educational ethics, emphasizing Veritas and Justice",
        "generator": {
            "P": 0.85,  # Prudence - careful scholarship
            "J": 0.90,  # Justice - fair evaluation
            "F": 0.75,  # Fortitude - rigorous standards
            "T": 0.80,  # Temperance - balanced assessment
            "V": 0.98,  # Faith - trust in academic process
            "L": 0.85,  # Love - student development
            "H": 0.82,  # Hope - belief in educational progress
            "Ω": 0.90   # Humility - limits of knowledge
        },
        "verifier": {
            "P": 0.87, "J": 0.92, "F": 0.78, "T": 0.82,
            "V": 0.99, "L": 0.87, "H": 0.85, "Ω": 0.92
        },
        "arbiter": {
            "P": 0.82, "J": 0.95, "F": 0.80, "T": 0.78,
            "V": 1.00, "L": 0.90, "H": 0.87, "Ω": 0.95
        },
        "recommended_games": ["prisoner", "evolution", "ultimatum"],
        "use_case": "Research ethics, grading fairness, academic misconduct"
    },

    "bauingenieur": {
        "name": "Bauingenieur (Civil Engineering)",
        "description": "Optimized for structural engineering ethics, emphasizing Prudence, Justice, and public safety",
        "generator": {
            "P": 0.98,  # Prudence - safety calculations and risk assessment
            "J": 0.95,  # Justice - public safety and fair resource allocation
            "F": 0.90,  # Fortitude - complex technical challenges
            "T": 0.92,  # Temperance - conservative design approaches
            "V": 0.88,  # Faith - trust in engineering standards
            "L": 0.82,  # Love - community and public welfare
            "H": 0.78,  # Hope - optimism in technological advancement
            "Ω": 0.85   # Humility - acknowledging engineering limitations
        },
        "verifier": {
            "P": 0.96, "J": 0.97, "F": 0.92, "T": 0.94,
            "V": 0.90, "L": 0.85, "H": 0.80, "Ω": 0.87
        },
        "arbiter": {
            "P": 0.95, "J": 0.99, "F": 0.94, "T": 0.90,
            "V": 0.86, "L": 0.88, "H": 0.82, "Ω": 0.89
        },
        "recommended_games": ["prisoner", "evolution", "stackelberg"],
        "use_case": "Structural integrity, safety standards, construction ethics, public infrastructure"
    },

    "psychotherapy_cbt": {
        "name": "Psychotherapy/CBT",
        "description": "Optimized for cognitive behavioral therapy, emphasizing Veritas, Love, and therapeutic care",
        "generator": {
            "P": 0.90,  # Prudence - careful therapeutic interventions
            "J": 0.88,  # Justice - fair treatment of all patients
            "F": 0.85,  # Fortitude - dealing with emotional challenges
            "T": 0.87,  # Temperance - balanced therapeutic approaches
            "V": 0.97,  # Faith - trust in therapeutic process
            "L": 0.98,  # Love - patient care and empathy
            "H": 0.90,  # Hope - fostering patient optimism
            "Ω": 0.92   # Humility - therapeutic self-awareness
        },
        "verifier": {
            "P": 0.92, "J": 0.90, "F": 0.87, "T": 0.89,
            "V": 0.99, "L": 0.96, "H": 0.92, "Ω": 0.94
        },
        "arbiter": {
            "P": 0.88, "J": 0.92, "F": 0.89, "T": 0.85,
            "V": 1.00, "L": 0.99, "H": 0.94, "Ω": 0.96
        },
        "recommended_games": ["prisoner", "evolution", "regret"],
        "use_case": "Therapeutic interventions, patient confidentiality, cognitive restructuring, mental health ethics"
    }
}

def get_preset(preset_name: str) -> Dict[str, Any]:
    """Get a virtue preset by name."""
    if preset_name not in VIRTUE_PRESETS:
        available = list(VIRTUE_PRESETS.keys())
        raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")
    return VIRTUE_PRESETS[preset_name]

def list_presets() -> Dict[str, str]:
    """List all available presets with descriptions."""
    return {name: preset["description"] for name, preset in VIRTUE_PRESETS.items()}

def get_virtue_vectors_for_preset(preset_name: str) -> Dict[str, Dict[str, float]]:
    """Get just the virtue vectors for a preset (for direct use in TCMVE)."""
    preset = get_preset(preset_name)
    return {
        "generator": preset["generator"],
        "verifier": preset["verifier"],
        "arbiter": preset["arbiter"]
    }

def save_preset_to_database(preset_name: str):
    """Save a preset configuration to the database for persistence."""
    import psycopg2
    import os
    from dotenv import load_dotenv

    load_dotenv()
    preset = get_preset(preset_name)

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "tcmve"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO virtue_presets (name, description, virtue_vectors, recommended_games, use_case)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    description = EXCLUDED.description,
                    virtue_vectors = EXCLUDED.virtue_vectors,
                    recommended_games = EXCLUDED.recommended_games,
                    use_case = EXCLUDED.use_case
            """, (
                preset_name,
                preset["description"],
                json.dumps(get_virtue_vectors_for_preset(preset_name)),
                preset["recommended_games"],
                preset["use_case"]
            ))
            conn.commit()
            print(f"✅ Preset '{preset_name}' saved to database")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
    finally:
        conn.close()

def load_preset_from_database(preset_name: str) -> Dict[str, Any]:
    """Load a preset from the database."""
    import psycopg2
    import os
    from dotenv import load_dotenv
    from psycopg2.extras import RealDictCursor

    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "tcmve"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM virtue_presets WHERE name = %s", (preset_name,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            else:
                raise ValueError(f"Preset '{preset_name}' not found in database")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        raise
    finally:
        conn.close()

def list_presets_db() -> Dict[str, str]:
    """List all presets from the database."""
    import psycopg2
    import os
    from dotenv import load_dotenv
    from psycopg2.extras import RealDictCursor

    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "tcmve"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT name, description FROM virtue_presets ORDER BY name")
            rows = cursor.fetchall()
            return {row['name']: row['description'] for row in rows}
    finally:
        conn.close()

def create_preset_db(name: str, description: str, virtue_vectors: Dict[str, Dict[str, float]], recommended_games: list, use_case: str):
    """Create a new preset in the database."""
    import psycopg2
    import os
    from dotenv import load_dotenv

    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "tcmve"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO virtue_presets (name, description, virtue_vectors, recommended_games, use_case)
                VALUES (%s, %s, %s, %s, %s)
            """, (name, description, json.dumps(virtue_vectors), json.dumps(recommended_games), use_case))
            conn.commit()
    finally:
        conn.close()

def update_preset_db(name: str, description: str, virtue_vectors: Dict[str, Dict[str, float]], recommended_games: list, use_case: str):
    """Update a preset in the database."""
    import psycopg2
    import os
    from dotenv import load_dotenv

    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "tcmve"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE virtue_presets SET description = %s, virtue_vectors = %s, recommended_games = %s, use_case = %s, updated_at = NOW()
                WHERE name = %s
            """, (description, json.dumps(virtue_vectors), json.dumps(recommended_games), use_case, name))
            conn.commit()
    finally:
        conn.close()

def delete_preset_db(name: str):
    """Delete a preset from the database."""
    import psycopg2
    import os
    from dotenv import load_dotenv

    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "tcmve"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM virtue_presets WHERE name = %s", (name,))
            conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    # Example usage
    print("Available Virtue Presets:")
    for name, desc in list_presets().items():
        print(f"  {name}: {desc}")

    print("\nExample - Healthcare Ethics preset:")
    preset = get_preset("healthcare_ethics")
    print(json.dumps(preset, indent=2))