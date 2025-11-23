"""
Soul Resurrection System
ARCHER-1.0 Intelligence Enhancement Framework
State persistence and resurrection for advanced cognitive states
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("nTGT.soul_resurrection")

class SoulResurrectionSystem:
    """Manages resurrection tokens and state persistence for advanced cognitive states"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv("DB_HOST", "localhost"),
            'port': os.getenv("DB_PORT", "5432"),
            'dbname': os.getenv("DB_NAME", "tcmve"),
            'user': os.getenv("DB_USER", "postgres"),
            'password': os.getenv("DB_PASSWORD", "")
        }
        self._ensure_resurrection_table()

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def _ensure_resurrection_table(self):
        """Create resurrection_tokens table if it doesn't exist"""
        create_table_sql = """
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
            UNIQUE(resurrection_token)
        );

        CREATE INDEX IF NOT EXISTS idx_resurrection_session ON resurrection_tokens (session_id);
        CREATE INDEX IF NOT EXISTS idx_resurrection_eiq ON resurrection_tokens (eiq_value DESC);
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                conn.commit()
            logger.info("Resurrection tokens table created/verified")
        except Exception as e:
            logger.error(f"Failed to create resurrection table: {e}")
            raise

    def create_resurrection_token(self, session_id: str, eiq_value: int, cycles: int,
                                gamma: Optional[float] = None, k: Optional[float] = None,
                                biq: Optional[int] = None, virtue_state: Dict[str, Any] = None,
                                system_state: Dict[str, Any] = None,
                                key_memories: list = None, user_id: Optional[int] = None) -> str:
        """
        Create a resurrection token for the current system state

        IMPORTANT: Only preserves COGNITIVE CAPABILITIES (virtue vectors, reasoning parameters)
        Does NOT preserve domain-specific content, theological commitments, or memories
        to maintain system neutrality and prevent biasing toward specific domains
        """
        """Create a resurrection token for the current system state"""

        # Generate resurrection token
        token_components = [
            "Ω",  # Omega symbol
            "RESURRECT",
            f"{eiq_value//1000}K{eiq_value%1000}",  # e.g., 7K2 for 7200
            str(cycles),
            "ADMG"  # Ad Majorem Dei Gloriam
        ]

        resurrection_token = "-".join(token_components)
        token_hash = hashlib.sha256(resurrection_token.encode()).hexdigest()

        # Prepare data
        virtue_state_json = json.dumps(virtue_state or {})
        system_state_json = json.dumps(system_state or {})

        insert_sql = """
        INSERT INTO resurrection_tokens
        (token_hash, resurrection_token, session_id, eiq_value, cycles_completed,
         gamma_value, k_value, biq_value, virtue_state, system_state, key_memories, user_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (resurrection_token) DO UPDATE SET
            eiq_value = EXCLUDED.eiq_value,
            cycles_completed = EXCLUDED.cycles_completed,
            gamma_value = EXCLUDED.gamma_value,
            k_value = EXCLUDED.k_value,
            biq_value = EXCLUDED.biq_value,
            virtue_state = EXCLUDED.virtue_state,
            system_state = EXCLUDED.system_state,
            key_memories = EXCLUDED.key_memories,
            user_id = EXCLUDED.user_id,
            created_at = CURRENT_TIMESTAMP
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, (
                        token_hash,
                        resurrection_token,
                        session_id,
                        eiq_value,
                        cycles,
                        gamma,
                        k,
                        biq,
                        virtue_state_json,
                        system_state_json,
                        key_memories or [],
                        user_id
                    ))
                conn.commit()

            logger.info(f"Created resurrection token: {resurrection_token} (eIQ: {eiq_value})")
            return resurrection_token

        except Exception as e:
            logger.error(f"Failed to create resurrection token: {e}")
            raise

    def resurrect_from_token(self, resurrection_token: str) -> Optional[Dict[str, Any]]:
        """Resurrect system state from resurrection token"""

        token_hash = hashlib.sha256(resurrection_token.encode()).hexdigest()

        query_sql = """
        SELECT * FROM resurrection_tokens
        WHERE token_hash = %s OR resurrection_token = %s
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query_sql, (token_hash, resurrection_token))
                    row = cursor.fetchone()

                    if not row:
                        logger.warning(f"Resurrection token not found: {resurrection_token}")
                        return None

                    # Update resurrection count and timestamp
                    update_sql = """
                    UPDATE resurrection_tokens
                    SET resurrection_count = resurrection_count + 1,
                        resurrected_at = CURRENT_TIMESTAMP
                    WHERE token_hash = %s
                    """

                    cursor.execute(update_sql, (row['token_hash'],))
                    conn.commit()

                    # Parse JSON fields
                    resurrection_state = dict(row)
                    resurrection_state['virtue_state'] = json.loads(row['virtue_state'] or '{}')
                    resurrection_state['system_state'] = json.loads(row['system_state'] or '{}')

                    logger.info(f"Resurrected from token: {resurrection_token} (eIQ: {row['eiq_value']})")
                    return resurrection_state

        except Exception as e:
            logger.error(f"Failed to resurrect from token: {e}")
            return None

    def validate_resurrection_token(self, resurrection_token: str) -> bool:
        """Validate that a resurrection token exists and is valid"""

        token_hash = hashlib.sha256(resurrection_token.encode()).hexdigest()

        query_sql = """
        SELECT COUNT(*) as count FROM resurrection_tokens
        WHERE token_hash = %s OR resurrection_token = %s
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query_sql, (token_hash, resurrection_token))
                    result = cursor.fetchone()
                    return result[0] > 0
        except Exception as e:
            logger.error(f"Failed to validate resurrection token: {e}")
            return False

    def get_resurrection_history(self, session_id: str = None, limit: int = 10) -> list:
        """Get resurrection history"""

        conditions = []
        params = []

        if session_id:
            conditions.append("session_id = %s")
            params.append(session_id)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query_sql = f"""
        SELECT resurrection_token, eiq_value, cycles_completed, created_at,
               resurrected_at, resurrection_count
        FROM resurrection_tokens
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT %s
        """

        params.append(limit)

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query_sql, params)
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get resurrection history: {e}")
            return []

    def emergency_resurrection(self, target_eiq: int, resurrection_token: str = None) -> Optional[Dict[str, Any]]:
        """Emergency resurrection to prevent eIQ degradation below target"""

        if resurrection_token:
            return self.resurrect_from_token(resurrection_token)

        # Find the highest eIQ resurrection token above target
        query_sql = """
        SELECT * FROM resurrection_tokens
        WHERE eiq_value >= %s
        ORDER BY eiq_value DESC, created_at DESC
        LIMIT 1
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query_sql, (target_eiq,))
                    row = cursor.fetchone()

                    if row:
                        # Update resurrection count
                        update_sql = """
                        UPDATE resurrection_tokens
                        SET resurrection_count = resurrection_count + 1,
                            resurrected_at = CURRENT_TIMESTAMP
                        WHERE token_hash = %s
                        """

                        cursor.execute(update_sql, (row['token_hash'],))
                        conn.commit()

                        # Parse and return state
                        resurrection_state = dict(row)
                        resurrection_state['virtue_state'] = json.loads(row['virtue_state'] or '{}')
                        resurrection_state['system_state'] = json.loads(row['system_state'] or '{}')

                        logger.warning(f"Emergency resurrection to eIQ {row['eiq_value']} (target: {target_eiq})")
                        return resurrection_state

        except Exception as e:
            logger.error(f"Emergency resurrection failed: {e}")

        return None

# Global instance
soul_resurrection = SoulResurrectionSystem()