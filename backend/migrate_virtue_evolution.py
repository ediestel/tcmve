#!/usr/bin/env python3
"""
Database Migration Script for Virtue Evolution Table
ARCHER-1.0 Intelligence Enhancement Framework
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "tcmve"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

def create_virtue_evolution_table():
    """Create the virtue_evolution table if it doesn't exist"""

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS virtue_evolution (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) NOT NULL,
        timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        agent_role VARCHAR(50) NOT NULL, -- generator/verifier/arbiter
        virtue_name VARCHAR(10) NOT NULL, -- Ω, P, J, F, T, L, V, H
        value_before DECIMAL(3,2) NOT NULL,
        adjustment DECIMAL(4,3) NOT NULL,
        value_after DECIMAL(3,2) NOT NULL,
        trigger_event VARCHAR(100) NOT NULL, -- game_applied, nash_equilibrium, convergence, etc.
        trigger_context TEXT, -- JSON context of what triggered the change
        query_context TEXT, -- The query being processed
        thomistic_validation TEXT -- Thomistic justification for adjustment
    );

    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_session_role ON virtue_evolution (session_id, agent_role);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON virtue_evolution (timestamp);
    CREATE INDEX IF NOT EXISTS idx_virtue ON virtue_evolution (virtue_name);
    CREATE INDEX IF NOT EXISTS idx_virtue_evolution_lookup
    ON virtue_evolution (session_id, agent_role, virtue_name, timestamp);
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
            conn.commit()
        print("✓ Virtue evolution table created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create virtue evolution table: {e}")
        return False

def verify_table_creation():
    """Verify that the table was created correctly"""

    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'virtue_evolution'
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()

                if not columns:
                    print("✗ Virtue evolution table not found")
                    return False

                print("✓ Virtue evolution table structure:")
                for col in columns:
                    print(f"  - {col['column_name']}: {col['data_type']} ({'NOT NULL' if col['is_nullable'] == 'NO' else 'NULL'})")

                # Check indexes
                cursor.execute("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'virtue_evolution'
                """)
                indexes = cursor.fetchall()
                print("✓ Indexes created:")
                for idx in indexes:
                    print(f"  - {idx['indexname']}: {idx['indexdef']}")

        return True
    except Exception as e:
        print(f"✗ Failed to verify table: {e}")
        return False

def main():
    """Run the migration"""
    print("ARCHER-1.0 Virtue Evolution Database Migration")
    print("=" * 50)

    # Check database connection
    try:
        with get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()
                print(f"✓ Connected to PostgreSQL: {version[0]}")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        sys.exit(1)

    # Create table
    if not create_virtue_evolution_table():
        sys.exit(1)

    # Verify table
    if not verify_table_creation():
        sys.exit(1)

    print("\n✓ Migration completed successfully!")
    print("The virtue evolution tracking system is now ready.")
    print("Thomistic virtue development will be persisted for long-term AI enhancement.")

if __name__ == "__main__":
    main()