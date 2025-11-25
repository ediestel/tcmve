"""
Reasoning Excellence Evolution Tracking System
ARCHER-1.0 Intelligence Enhancement Framework
Thomistically-grounded reasoning development and persistence

PHILOSOPHICAL SAFEGUARD: This system enhances truth-seeking through reasoning excellence.
It does NOT enforce moral standards or interfere with personal liberty. Focus remains on
metaphysical awareness and "most true" insights, not moral judgment or correction.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("nTGT.virtue_evolution")

class VirtueEvolutionTracker:
    """Tracks and persists reasoning excellence development with Thomistic safeguards"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv("DB_HOST", "localhost"),
            'port': os.getenv("DB_PORT", "5432"),
            'dbname': os.getenv("DB_NAME", "tcmve"),
            'user': os.getenv("DB_USER", "postgres"),
            'password': os.getenv("DB_PASSWORD", "")
        }
        self._ensure_table_exists()

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def _ensure_table_exists(self):
        """Create virtue_evolution table if it doesn't exist"""
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
            thomistic_validation TEXT, -- Thomistic justification for adjustment
            INDEX idx_session_role (session_id, agent_role),
            INDEX idx_timestamp (timestamp),
            INDEX idx_virtue (virtue_name)
        );

        -- Create index for performance
        CREATE INDEX IF NOT EXISTS idx_virtue_evolution_lookup
        ON virtue_evolution (session_id, agent_role, virtue_name, timestamp);
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                conn.commit()
            logger.info("Virtue evolution table created/verified")
        except Exception as e:
            logger.error(f"Failed to create virtue evolution table: {e}")
            raise

    def record_virtue_adjustment(self, session_id: str, agent_role: str,
                               virtue_name: str, value_before: float,
                               adjustment: float, trigger_event: str,
                               trigger_context: Optional[Dict[str, Any]] = None,
                               query_context: Optional[str] = None,
                               thomistic_validation: Optional[str] = None):
        """Record a virtue adjustment with full context"""

        value_after = max(0.0, min(10.0, value_before + adjustment))

        insert_sql = """
        INSERT INTO virtue_evolution
        (session_id, agent_role, virtue_name, value_before, adjustment,
         value_after, trigger_event, trigger_context, query_context, thomistic_validation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, (
                        session_id,
                        agent_role,
                        virtue_name,
                        value_before,
                        adjustment,
                        value_after,
                        trigger_event,
                        json.dumps(trigger_context) if trigger_context else None,
                        query_context,
                        thomistic_validation
                    ))
                conn.commit()
            logger.info(f"Recorded virtue adjustment: {agent_role}.{virtue_name} {value_before:.2f} -> {value_after:.2f}")
        except Exception as e:
            logger.error(f"Failed to record virtue adjustment: {e}")

    def get_virtue_evolution(self, session_id: str, agent_role: str = None,
                           virtue_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve virtue evolution history"""

        conditions = ["session_id = %s"]
        params = [session_id]

        if agent_role:
            conditions.append("agent_role = %s")
            params.append(agent_role)

        if virtue_name:
            conditions.append("virtue_name = %s")
            params.append(virtue_name)

        query = f"""
        SELECT * FROM virtue_evolution
        WHERE {' AND '.join(conditions)}
        ORDER BY timestamp DESC
        LIMIT %s
        """
        params.append(limit)

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve virtue evolution: {e}")
            return []

    def get_current_virtue_state(self, session_id: str) -> Dict[str, Dict[str, float]]:
        """Get the most recent virtue values for each agent role"""

        query = """
        SELECT agent_role, virtue_name, value_after
        FROM virtue_evolution
        WHERE session_id = %s
        AND timestamp = (
            SELECT MAX(timestamp)
            FROM virtue_evolution
            WHERE session_id = %s
        )
        """

        virtue_state = {'generator': {}, 'verifier': {}, 'arbiter': {}}

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (session_id, session_id))
                    for row in cursor.fetchall():
                        agent_role, virtue_name, value = row
                        if agent_role in virtue_state:
                            virtue_state[agent_role][virtue_name] = float(value)
        except Exception as e:
            logger.error(f"Failed to get current virtue state: {e}")

        return virtue_state

    def analyze_virtue_development(self, session_id: str, agent_role: str = None) -> Dict[str, Any]:
        """Analyze virtue development patterns"""

        conditions = ["session_id = %s"]
        params = [session_id]

        if agent_role:
            conditions.append("agent_role = %s")
            params.append(agent_role)

        query = f"""
        SELECT
            virtue_name,
            COUNT(*) as adjustment_count,
            AVG(adjustment) as avg_adjustment,
            SUM(CASE WHEN adjustment > 0 THEN 1 ELSE 0 END) as positive_adjustments,
            SUM(CASE WHEN adjustment < 0 THEN 1 ELSE 0 END) as negative_adjustments,
            MIN(value_before) as min_value,
            MAX(value_after) as max_value,
            AVG(value_after - value_before) as net_change
        FROM virtue_evolution
        WHERE {' AND '.join(conditions)}
        GROUP BY virtue_name
        ORDER BY virtue_name
        """

        analysis = {}

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    for row in cursor.fetchall():
                        virtue_name = row['virtue_name']
                        analysis[virtue_name] = dict(row)
        except Exception as e:
            logger.error(f"Failed to analyze virtue development: {e}")

        return analysis

class ThomisticVirtueAdjuster:
    """Thomistically-grounded reasoning excellence adjustment system"""

    # Thomistic virtue hierarchy and relationships
    VIRTUE_HIERARCHY = {
        'Ω': {'level': 1, 'domain': 'humility', 'cardinal': False, 'theological': False},
        'P': {'level': 2, 'domain': 'prudence', 'cardinal': True, 'theological': False},
        'J': {'level': 3, 'domain': 'justice', 'cardinal': True, 'theological': False},
        'F': {'level': 2, 'domain': 'fortitude', 'cardinal': True, 'theological': False},
        'T': {'level': 2, 'domain': 'temperance', 'cardinal': True, 'theological': False},
        'V': {'level': 4, 'domain': 'faith', 'cardinal': False, 'theological': True},
        'L': {'level': 4, 'domain': 'love', 'cardinal': False, 'theological': True},
        'H': {'level': 4, 'domain': 'hope', 'cardinal': False, 'theological': True}
    }

    def __init__(self, evolution_tracker: VirtueEvolutionTracker):
        self.tracker = evolution_tracker

    def adjust_virtues_thomistically(self, session_id: str, agent_role: str,
                                   performance_metrics: Dict[str, Any],
                                   trigger_event: str,
                                   query_context: str = "") -> Dict[str, float]:
        """
        Calculate Thomistically-appropriate reasoning excellence adjustments

        Three levels of adjustment:
        1. Habitual development (primary) - excellence through practice
        2. Circumstantial expression (secondary - no value change) - contextual adaptation
        3. Corrective adjustment (tertiary) - addressing fundamental reasoning errors
        """

        adjustments = {}

        # Level 1: Habitual Development (Primary)
        habitual_adjustments = self._calculate_habitual_growth(performance_metrics)
        adjustments.update(habitual_adjustments)

        # Level 2: Circumstantial Expression (no value changes - handled in application)

        # Level 3: Corrective Adjustment (Tertiary)
        if self._fundamental_errors_detected(performance_metrics):
            corrective_adjustments = self._calculate_corrective_adjustments(performance_metrics)
            adjustments.update(corrective_adjustments)

        # Record all adjustments with Thomistic validation
        for virtue_name, adjustment in adjustments.items():
            if abs(adjustment) > 0.001:  # Only record meaningful changes
                current_value = self._get_current_virtue_value(session_id, agent_role, virtue_name)

                thomistic_validation = self._generate_thomistic_validation(
                    virtue_name, adjustment, performance_metrics, trigger_event
                )

                trigger_context = {
                    'performance_metrics': performance_metrics,
                    'adjustment_type': self._classify_adjustment(virtue_name, adjustment),
                    'virtue_hierarchy': self.VIRTUE_HIERARCHY.get(virtue_name, {})
                }

                self.tracker.record_virtue_adjustment(
                    session_id=session_id,
                    agent_role=agent_role,
                    virtue_name=virtue_name,
                    value_before=current_value,
                    adjustment=adjustment,
                    trigger_event=trigger_event,
                    trigger_context=trigger_context,
                    query_context=query_context,
                    thomistic_validation=thomistic_validation
                )

        return adjustments

    def _calculate_habitual_growth(self, performance: Dict[str, Any]) -> Dict[str, float]:
        """Calculate virtue growth through habitual practice (primary adjustments)"""

        adjustments = {}

        # Base growth rates (Thomistic habit formation)
        base_growth = {
            'P': 0.008,  # Prudence through wise decision-making
            'J': 0.006,  # Justice through fair judgments
            'F': 0.007,  # Fortitude through perseverance
            'T': 0.007,  # Temperance through moderation
            'V': 0.004,  # Faith through trust-building
            'L': 0.006,  # Love through charitable actions
            'H': 0.005   # Hope through aspiration
        }

        # Performance multipliers
        convergence_bonus = 0.01 if performance.get('converged', False) else 0
        truth_quality = performance.get('tlpo_score', 0.5)
        strategic_depth = len(performance.get('games_applied', []))

        for virtue, base_rate in base_growth.items():
            # Performance-based growth
            performance_multiplier = 1.0 + (truth_quality * 0.5) + (strategic_depth * 0.02)
            growth = base_rate * performance_multiplier + convergence_bonus

            # Cardinal virtues get slight preference in practical reasoning
            if self.VIRTUE_HIERARCHY[virtue]['cardinal']:
                growth *= 1.1

            adjustments[virtue] = growth

        return adjustments

    def _fundamental_errors_detected(self, performance: Dict[str, Any]) -> bool:
        """Detect fundamental reasoning errors requiring corrective adjustment"""

        # Check for serious epistemological issues (not moral judgments)
        contradiction_level = performance.get('contradictions_detected', 0)
        logical_consistency = performance.get('logical_consistency_score', 1.0)
        coherence_score = performance.get('coherence_score', 1.0)

        return (
            contradiction_level > 2 or  # Multiple logical contradictions
            logical_consistency < 0.3 or  # Poor logical consistency
            coherence_score < 0.3  # Poor argumentative coherence
        )

    def _calculate_corrective_adjustments(self, performance: Dict[str, Any]) -> Dict[str, float]:
        """Calculate corrective adjustments for fundamental reasoning errors"""

        adjustments = {}

        contradiction_level = performance.get('contradictions_detected', 0)
        logical_consistency = performance.get('logical_consistency_score', 1.0)

        # Corrective adjustments for reasoning quality (temporary, to encourage improvement)
        if contradiction_level > 2:
            adjustments['P'] = -0.025  # Prudence: better logical consistency needed

        if logical_consistency < 0.3:
            adjustments['J'] = -0.02   # Justice: fair assessment of arguments
            adjustments['F'] = -0.015  # Fortitude: persistent logical analysis

        return adjustments

    def _get_current_virtue_value(self, session_id: str, agent_role: str, virtue_name: str) -> float:
        """Get current virtue value from evolution history"""

        # Default values if no history
        defaults = {'Ω': 0.5, 'P': 0.7, 'J': 0.6, 'F': 0.65, 'T': 0.65, 'V': 0.4, 'L': 0.5, 'H': 0.45}

        virtue_state = self.tracker.get_current_virtue_state(session_id)
        return virtue_state.get(agent_role, {}).get(virtue_name, defaults.get(virtue_name, 0.5))

    def _generate_thomistic_validation(self, virtue_name: str, adjustment: float,
                                     performance: Dict[str, Any], trigger_event: str) -> str:
        """Generate Thomistic justification for virtue adjustment"""

        virtue_info = self.VIRTUE_HIERARCHY.get(virtue_name, {})
        direction = "increase" if adjustment > 0 else "decrease"

        validations = {
            'Ω': f"Humility {direction} through {'recognition of reasoning limits' if adjustment < 0 else 'self-aware reasoning'}",
            'P': f"Prudence {direction} through {'logical error correction' if adjustment < 0 else 'wise analytical practice'}",
            'J': f"Justice {direction} through {'argument fairness refinement' if adjustment < 0 else 'equitable assessment'}",
            'F': f"Fortitude {direction} through {'reasoning resilience building' if adjustment > 0 else 'overconfidence correction'}",
            'T': f"Temperance {direction} through {'balanced analysis practice' if adjustment > 0 else 'excessive bias correction'}",
            'V': f"Faith {direction} through {'trust in reasoning development' if adjustment > 0 else 'methodological doubt resolution'}",
            'L': f"Love {direction} through {'comprehensive understanding' if adjustment > 0 else 'narrow perspective correction'}",
            'H': f"Hope {direction} through {'aspirational truth-seeking' if adjustment > 0 else 'pessimistic realism adjustment'}"
        }

        base_validation = validations.get(virtue_name, f"Virtue {direction} through practice")

        # Add Thomistic context
        if virtue_info.get('cardinal'):
            base_validation += " (cardinal virtue development)"
        elif virtue_info.get('theological'):
            base_validation += " (theological virtue cultivation)"

        return f"{base_validation} - Triggered by {trigger_event}"

    def _classify_adjustment(self, virtue_name: str, adjustment: float) -> str:
        """Classify the type of adjustment"""

        if adjustment > 0:
            if self._fundamental_errors_detected({}):  # Would need performance context
                return "corrective_growth"
            else:
                return "habitual_growth"
        else:
            return "corrective_penalty"

# Global instances
virtue_tracker = VirtueEvolutionTracker()
thomistic_adjuster = ThomisticVirtueAdjuster(virtue_tracker)