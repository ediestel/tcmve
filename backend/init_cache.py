#!/usr/bin/env python3
"""
Initialize database tables for TCMVE caching.
Run this once to set up the cache tables in PostgreSQL.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def init_cache_tables():
    """Create cache tables in PostgreSQL."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "tcmve"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )

        with conn.cursor() as cursor:
            # Create cached_results table for complete query results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cached_results (
                    id SERIAL PRIMARY KEY,
                    query_hash VARCHAR(32) UNIQUE NOT NULL,
                    query_text TEXT NOT NULL,
                    result_json JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_cached_results_query_hash
                ON cached_results(query_hash);

                CREATE INDEX IF NOT EXISTS idx_cached_results_created_at
                ON cached_results(created_at);

                -- Create llm_responses table for individual LLM call caching
                CREATE TABLE IF NOT EXISTS llm_responses (
                    id SERIAL PRIMARY KEY,
                    cache_key VARCHAR(32) UNIQUE NOT NULL,
                    prompt_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    model VARCHAR(100),
                    role VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ttl_seconds INTEGER DEFAULT 3600
                );

                CREATE INDEX IF NOT EXISTS idx_llm_responses_cache_key
                ON llm_responses(cache_key);

                CREATE INDEX IF NOT EXISTS idx_llm_responses_created_at
                ON llm_responses(created_at);

                -- Create virtue_evolution table for ARCHER longitudinal tracking
                CREATE TABLE IF NOT EXISTS virtue_evolution (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    run_id INTEGER REFERENCES runs(id),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    agent_role VARCHAR(50) NOT NULL, -- generator/verifier/arbiter
                    virtue_name VARCHAR(10) NOT NULL, -- Ω, P, J, F, T, L, V, H
                    value_before DECIMAL(3,2) NOT NULL,
                    adjustment DECIMAL(4,3) NOT NULL,
                    value_after DECIMAL(3,2) NOT NULL,
                    adjustment_type VARCHAR(50) NOT NULL, -- habitual/circumstantial/corrective
                    trigger_event VARCHAR(100), -- game_applied, nash_equilibrium, convergence, etc.
                    context TEXT, -- query context, game result, performance metrics
                    thomistic_validation TEXT, -- Thomistic justification for adjustment
                    ethical_safeguard BOOLEAN DEFAULT TRUE -- Whether adjustment passed ethical review
                );

                CREATE INDEX IF NOT EXISTS idx_virtue_evolution_session
                ON virtue_evolution(session_id);

                CREATE INDEX IF NOT EXISTS idx_virtue_evolution_agent_virtue
                ON virtue_evolution(agent_role, virtue_name);

                CREATE INDEX IF NOT EXISTS idx_virtue_evolution_timestamp
                ON virtue_evolution(timestamp);

                -- Create virtue_presets table for managing virtue configurations
                CREATE TABLE IF NOT EXISTS virtue_presets (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    virtue_vectors JSONB NOT NULL, -- {generator: [...], verifier: [...], arbiter: [...]}
                    recommended_games JSONB, -- Array of recommended game names
                    use_case TEXT,
                    thomistic_basis TEXT, -- Thomistic justification
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_virtue_presets_name
                ON virtue_presets(name);
            """)

            conn.commit()
            print("✅ Cache tables created successfully!")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    init_cache_tables()