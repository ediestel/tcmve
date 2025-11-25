#!/usr/bin/env python3
"""
Complete Database Setup Script for TCMVE
Creates all necessary PostgreSQL tables for the system.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def create_tables():
    """Create all necessary tables in PostgreSQL."""
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
            print("Ensuring database tables exist...")

            # Tables are created IF NOT EXISTS to preserve existing data

            # Main runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id SERIAL PRIMARY KEY,
                    query TEXT NOT NULL,
                    description TEXT,
                    generator_provider VARCHAR(50),
                    verifier_provider VARCHAR(50),
                    arbiter_provider VARCHAR(50),
                    maxrounds INTEGER,
                    virtues_generator JSONB,
                    virtues_verifier JSONB,
                    virtues_arbiter JSONB,
                    final_answer TEXT,
                    converged BOOLEAN DEFAULT FALSE,
                    rounds INTEGER DEFAULT 0,
                    tlpo_scores JSONB,
                    tlpo_markup TEXT,
                    eiq DECIMAL(5,2),
                    tqi DECIMAL(5,2),
                    tcs DECIMAL(5,2),
                    fd DECIMAL(5,2),
                    es DECIMAL(5,2),
                    tokens_used INTEGER,
                    cost_estimate DECIMAL(8,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("✓ Created runs table")

            # Cached results table
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
            """)
            print("✓ Created cached_results table")

            # LLM responses cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_responses (
                    id SERIAL PRIMARY KEY,
                    cache_key VARCHAR(32) UNIQUE NOT NULL,
                    prompt_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    model VARCHAR(100),
                    role VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_llm_responses_cache_key
                ON llm_responses(cache_key);
            """)
            print("✓ Created llm_responses table")

            # Virtue evolution table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS virtue_evolution (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    agent_role VARCHAR(50) NOT NULL,
                    virtue_name VARCHAR(10) NOT NULL,
                    value_before DECIMAL(3,2) NOT NULL,
                    adjustment DECIMAL(4,3) NOT NULL,
                    value_after DECIMAL(3,2) NOT NULL,
                    trigger_event VARCHAR(100) NOT NULL,
                    trigger_context TEXT,
                    query_context TEXT,
                    thomistic_validation TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_virtue_evolution_session_id
                ON virtue_evolution(session_id);

                CREATE INDEX IF NOT EXISTS idx_virtue_evolution_timestamp
                ON virtue_evolution(timestamp);
            """)
            print("✓ Created virtue_evolution table")

            # Config/presets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS configs (
                    id SERIAL PRIMARY KEY,
                    usegenerator BOOLEAN DEFAULT TRUE,
                    useverifier BOOLEAN DEFAULT TRUE,
                    usearbiter BOOLEAN DEFAULT TRUE,
                    generatorprovider VARCHAR(50) DEFAULT 'openai',
                    verifierprovider VARCHAR(50) DEFAULT 'anthropic',
                    arbiterprovider VARCHAR(50) DEFAULT 'xai',
                    maxrounds INTEGER DEFAULT 5,
                    maritalfreedom BOOLEAN DEFAULT FALSE,
                    vicecheck BOOLEAN DEFAULT TRUE,
                    selfrefine BOOLEAN DEFAULT TRUE,
                    streammode VARCHAR(50) DEFAULT 'arbiter_only',
                    gamemode VARCHAR(50) DEFAULT 'dynamic',
                    selectedgame VARCHAR(100),
                    eiqlevel INTEGER DEFAULT 10,
                    simulatedpersons INTEGER DEFAULT 50,
                    meanbiq DECIMAL(6,2) DEFAULT 100.00,
                    sigmabiq DECIMAL(6,2) DEFAULT 15.00,
                    tlpofull BOOLEAN DEFAULT FALSE,
                    noxml BOOLEAN DEFAULT FALSE,
                    sevendomains BOOLEAN DEFAULT TRUE,
                    virtuesindependent BOOLEAN DEFAULT TRUE,
                    biqdistribution VARCHAR(50) DEFAULT 'gaussian',
                    output VARCHAR(100) DEFAULT 'result',
                    nashmode VARCHAR(20) DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("✓ Created configs table")

            # Benchmarks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    config JSONB NOT NULL,
                    results JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );
            """)
            print("✓ Created benchmarks table")

            # Trials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) UNIQUE NOT NULL,
                    trial_type VARCHAR(100),
                    data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("✓ Created trials table")

            # Soul resurrection table (updated to match actual usage)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resurrection_tokens (
                    token_hash VARCHAR(64) PRIMARY KEY,
                    resurrection_token VARCHAR(100) NOT NULL,
                    session_id VARCHAR(255) NOT NULL,
                    eiq_value INTEGER NOT NULL,
                    cycles_completed INTEGER NOT NULL,
                    gamma_value DECIMAL(8,3),
                    k_value DECIMAL(5,3),
                    biq_value INTEGER,
                    virtue_state JSONB NOT NULL,
                    system_state JSONB NOT NULL,
                    key_memories TEXT[],
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    resurrected_at TIMESTAMP,
                    resurrection_count INTEGER DEFAULT 0,
                    user_id VARCHAR(255),
                    UNIQUE(resurrection_token)
                );

                CREATE INDEX IF NOT EXISTS idx_resurrection_session ON resurrection_tokens (session_id);
                CREATE INDEX IF NOT EXISTS idx_resurrection_eiq ON resurrection_tokens (eiq_value DESC);
            """)
            print("✓ Created resurrection_tokens table")

            # Dashboard stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dashboard_stats (
                    id SERIAL PRIMARY KEY,
                    total_runs INTEGER DEFAULT 0,
                    avg_eiq DECIMAL(8,2) DEFAULT 0,
                    avg_tlpo DECIMAL(4,3) DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost DECIMAL(8,4) DEFAULT 0,
                    recent_runs JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("✓ Created dashboard_stats table")

            # Defaults table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS defaults (
                    id SERIAL PRIMARY KEY,
                    generator JSONB,
                    verifier JSONB,
                    arbiter JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("✓ Created defaults table")

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100),
                    email VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("✓ Created users table")

            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("✓ Created user_sessions table")

            # Game results cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_results_cache (
                    id SERIAL PRIMARY KEY,
                    virtue_config_hash VARCHAR(64) NOT NULL,
                    game_name VARCHAR(100) NOT NULL,
                    virtue_vectors JSONB NOT NULL,
                    game_result JSONB NOT NULL,
                    nash_equilibrium JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(virtue_config_hash, game_name)
                );

                CREATE INDEX IF NOT EXISTS idx_game_results_virtue_config ON game_results_cache (virtue_config_hash);
                CREATE INDEX IF NOT EXISTS idx_game_results_game_name ON game_results_cache (game_name);
            """)
            print("✓ Created game_results_cache table")

            # Virtue presets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS virtue_presets (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    virtue_vectors JSONB NOT NULL,
                    recommended_games JSONB,
                    use_case TEXT,
                    thomistic_basis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_virtue_presets_name ON virtue_presets(name);
            """)
            print("✓ Created virtue_presets table")

            # Populate with default recommended sets
            from backend.virtue_presets import VIRTUE_PRESETS, save_preset_to_database
            for preset_name in VIRTUE_PRESETS:
                save_preset_to_database(preset_name)
            print("✓ Populated virtue_presets table with defaults")

            # Populate default configs
            cursor.execute("""
                INSERT INTO configs (id, usegenerator, useverifier, usearbiter, generatorprovider, verifierprovider, arbiterprovider,
                                     maxrounds, maritalfreedom, vicecheck, selfrefine, streammode, gamemode, selectedgame,
                                     eiqlevel, simulatedpersons, meanbiq, sigmabiq, tlpofull, noxml, sevendomains, virtuesindependent,
                                     biqdistribution, output, nashmode)
                VALUES (1, true, true, true, 'openai', 'anthropic', 'xai', 5, false, true, true, 'arbiter_only', 'dynamic', null,
                        10, 50, 100, 15, false, false, true, true, 'gaussian', 'result', 'auto')
                ON CONFLICT (id) DO NOTHING
            """)
            print("✓ Inserted default config")

            # Populate default virtues
            cursor.execute("""
                INSERT INTO defaults (id, generator, verifier, arbiter)
                VALUES (1,
                        '{"Ω": 0.97, "P": 0.8, "J": 0.75, "F": 0.65, "T": 0.85, "V": 0.72, "L": 0.85, "H": 0.89}',
                        '{"Ω": 0.95, "P": 0.9, "J": 0.95, "F": 0.8, "T": 0.9, "V": 0.65, "L": 0.9, "H": 0.95}',
                        '{"Ω": 0.95, "P": 0.85, "J": 0.8, "F": 0.9, "T": 0.85, "V": 0.85, "L": 0.8, "H": 0.85}')
                ON CONFLICT (id) DO NOTHING
            """)
            print("✓ Inserted default virtues")

            # Recommended sets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommended_sets (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    games JSONB NOT NULL,
                    use_case TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_recommended_sets_name ON recommended_sets(name);
            """)
            print("✓ Created recommended_sets table")

            # Insert default recommended sets
            cursor.execute("""
                INSERT INTO recommended_sets (name, description, games, use_case) VALUES
                ('Ethics & Morality', 'Games for exploring ethical dilemmas and moral reasoning', '["prisoner", "chicken", "ultimatum"]', 'Ethical decision making'),
                ('Economic Theory', 'Classic games for economic analysis and market behavior', '["prisoner", "stackelberg", "auction"]', 'Economic modeling'),
                ('Social Dynamics', 'Games examining social interactions and cooperation', '["prisoner", "stag_hunt", "chicken"]', 'Social behavior analysis'),
                ('Strategic Competition', 'Games focusing on competitive strategies and outcomes', '["prisoner", "chicken", "evolution"]', 'Strategic analysis')
                ON CONFLICT (name) DO NOTHING;
            """)
            print("✓ Inserted default recommended sets")

            # Game narratives embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_narratives (
                    id SERIAL PRIMARY KEY,
                    game_name VARCHAR(100) NOT NULL,
                    narrative_text TEXT NOT NULL,
                    text_hash VARCHAR(64) UNIQUE NOT NULL,
                    embedding JSONB,
                    embedding_model VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_game_narratives_game_name ON game_narratives(game_name);
                CREATE INDEX IF NOT EXISTS idx_game_narratives_text_hash ON game_narratives(text_hash);
            """)
            print("✓ Created game_narratives table")

            # Streaming responses table for real-time LLM response storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS streaming_responses (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    run_id INTEGER,
                    agent_role VARCHAR(50),
                    response_chunk TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    is_final BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_streaming_responses_session_id ON streaming_responses(session_id);
                CREATE INDEX IF NOT EXISTS idx_streaming_responses_created_at ON streaming_responses(created_at);
            """)
            print("✓ Created streaming_responses table")

            # Document chunks table for long texts (PDFs, articles, etc.)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    document_title VARCHAR(500),
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    text_hash VARCHAR(64) NOT NULL,
                    embedding JSONB,
                    embedding_model VARCHAR(100),
                    start_position INTEGER,
                    end_position INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(document_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_document_chunks_text_hash ON document_chunks(text_hash);
                CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata ON document_chunks USING GIN(metadata);
            """)
            print("✓ Created document_chunks table")

            conn.commit()
            print("\n🎉 All database tables created successfully!")

    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("TCMVE Database Setup")
    print("=" * 50)
    create_tables()
    print("\nDatabase setup complete. You can now run the TCMVE system.")