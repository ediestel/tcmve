"""
Dynamic Game Selection for TCMVE
ARCHER-1.0 Intelligence Enhancement Framework
"""

from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger("nTGT.game_selector")

@dataclass
class GameRecommendation:
    """Game recommendation with priority and rationale"""
    game_name: str
    priority: float  # 0.0-1.0
    rationale: str
    virtue_alignment: Dict[str, float]  # Which virtues this game enhances

class DynamicGameSelector:
    """Dynamic game selection based on virtue profiles and query context"""

    def _select_games_by_embedding(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Use existing OpenAI text-embedding-3-large embeddings from database."""
        try:
            from .query_embeddings import find_similar_games
            similar_games = find_similar_games(query, top_k=top_k)
            return [(name, score) for name, score, _ in similar_games]
        except Exception as e:
            logger.warning(f"Embedding-based selection failed: {e}")
            return []

    # Game-Virtue Alignment Matrix
    GAME_VIRTUE_MATRIX = {
        "prisoner": {
            "P": 0.8,  # Prudence - strategic foresight
            "J": 0.9,  # Justice - fairness/cooperation
            "F": 0.6,  # Fortitude - resilience in dilemma
            "rationale": "Tests cooperation vs defection, enhances justice and prudence"
        },
        "auction": {
            "P": 0.9,  # Prudence - bidding strategy
            "T": 0.8,  # Temperance - controlled bidding
            "F": 0.7,  # Fortitude - persistence in competition
            "rationale": "Strategic resource allocation, enhances temperance and prudence"
        },
        "stackelberg": {
            "P": 0.9,  # Prudence - leader/follower dynamics
            "Ω": 0.8,  # Humility - recognizing leadership roles
            "F": 0.7,  # Fortitude - maintaining leadership
            "rationale": "Hierarchical strategy, enhances prudence and humility"
        },
        "evolution": {
            "F": 0.9,  # Fortitude - survival adaptation
            "H": 0.8,  # Hope - evolutionary optimism
            "V": 0.6,  # Faith - belief in adaptation
            "rationale": "Survival strategies, enhances fortitude and hope"
        },
        "regret_min": {
            "P": 0.8,  # Prudence - minimizing future regret
            "Ω": 0.9,  # Humility - learning from mistakes
            "T": 0.7,  # Temperance - balanced decision-making
            "rationale": "Learning from suboptimal choices, enhances humility"
        },
        "shadow_play": {
            "Ω": 0.8,  # Humility - shadow self-awareness
            "L": 0.7,  # Love - integrating shadow aspects
            "J": 0.6,  # Justice - fair self-assessment
            "rationale": "Shadow work integration, enhances humility and love"
        },
        "multiplay": {
            "L": 0.9,  # Love - multi-agent cooperation
            "J": 0.8,  # Justice - equitable outcomes
            "V": 0.7,  # Faith - trust in collective intelligence
            "rationale": "Multi-agent coordination, enhances love and justice"
        }
    }

    # Sequential Game Order (Thomistic progression)
    SEQUENTIAL_ORDER = [
        "prisoner",      # Foundation: Cooperation vs Competition
        "auction",       # Resource Management: Strategic Allocation
        "stackelberg",   # Hierarchy: Leadership Dynamics
        "regret_min",    # Learning: From Mistakes
        "evolution",     # Adaptation: Survival Strategies
        "shadow_play",   # Integration: Self-Knowledge
        "multiplay"      # Synthesis: Collective Intelligence
    ]

    def select_games_dynamic(self, virtue_vectors: Dict[str, Dict[str, float]],
                           query_context: str = "",
                           max_games: int = 3,
                           execution_mode: str = "sequential",
                           selection_mode: str = "rule_based") -> List[GameRecommendation]:
        """
        Dynamic game selection based on virtue profiles or semantic similarity

        Args:
            virtue_vectors: Dict with 'generator', 'verifier', 'arbiter' virtue arrays
            query_context: The query being analyzed
            max_games: Maximum number of games to select
            execution_mode: 'sequential' or 'parallel'
            selection_mode: 'rule_based' or 'embedding'

        Returns:
            List of GameRecommendation objects ordered by priority
        """

        if selection_mode == "embedding":
            # Use embedding-based selection
            top_games = self._select_games_by_embedding(query_context, top_k=max_games)
            if not top_games:
                # Fallback to rule-based if embedding fails
                logger.warning("Embedding selection failed, falling back to rule-based")
                selection_mode = "rule_based"
            else:
                # Convert to GameRecommendation format
                selected_games = []
                for game_name, similarity_score in top_games:
                    virtue_weights = self.GAME_VIRTUE_MATRIX.get(game_name, {})
                    recommendation = GameRecommendation(
                        game_name=game_name,
                        priority=min(similarity_score, 1.0),  # Cap at 1.0
                        rationale=f"Semantic similarity: {similarity_score:.3f}",
                        virtue_alignment={k: v for k, v in virtue_weights.items() if k != "rationale"}
                    )
                    selected_games.append(recommendation)

                logger.info(f"Embedding-selected games: {[g.game_name for g in selected_games]}")
                return selected_games

        # Rule-based selection (default)
        # Calculate average virtue profile across all agents
        avg_virtues = self._calculate_average_virtues(virtue_vectors)

        # Score each game based on virtue alignment
        game_scores = []
        for game_name, virtue_weights in self.GAME_VIRTUE_MATRIX.items():
            alignment_score = self._calculate_alignment_score(avg_virtues, virtue_weights)
            context_bonus = self._calculate_context_bonus(game_name, query_context)

            total_score = alignment_score + context_bonus

            recommendation = GameRecommendation(
                game_name=game_name,
                priority=min(total_score, 1.0),  # Cap at 1.0
                rationale=virtue_weights.get("rationale", ""),
                virtue_alignment={k: v for k, v in virtue_weights.items() if k != "rationale"}
            )
            game_scores.append((total_score, recommendation))

        # Sort by score descending
        game_scores.sort(key=lambda x: x[0], reverse=True)

        # Select top games, respecting sequential order if needed
        selected_games = []
        if execution_mode == "sequential":
            # For sequential: respect Thomistic progression order
            available_games = [rec for _, rec in game_scores[:max_games]]
            # Reorder according to SEQUENTIAL_ORDER
            ordered_games = []
            for game_name in self.SEQUENTIAL_ORDER:
                for rec in available_games:
                    if rec.game_name == game_name:
                        ordered_games.append(rec)
                        break
            selected_games = ordered_games[:max_games]
        else:
            # For parallel: take top scores
            selected_games = [rec for _, rec in game_scores[:max_games]]

        logger.info(f"Selected {len(selected_games)} games for {execution_mode} execution: "
                   f"{[g.game_name for g in selected_games]}")

        return selected_games

    def _calculate_average_virtues(self, virtue_vectors: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate average virtue values across all agents"""
        virtue_names = ['Ω', 'P', 'J', 'F', 'T', 'L', 'V', 'H']  # Humility, Prudence, Justice, Fortitude, Temperance, Love, Faith, Hope

        avg_virtues = {}
        for virtue in virtue_names:
            values = []
            for agent in ['generator', 'verifier', 'arbiter']:
                if agent in virtue_vectors and virtue in virtue_vectors[agent]:
                    values.append(virtue_vectors[agent][virtue])
            avg_virtues[virtue] = sum(values) / len(values) if values else 0.5

        return avg_virtues

    def _calculate_alignment_score(self, avg_virtues: Dict[str, float],
                                 game_weights: Dict[str, float]) -> float:
        """Calculate how well a game aligns with current virtue profile"""
        total_score = 0.0
        total_weight = 0.0

        for virtue, weight in game_weights.items():
            if virtue in avg_virtues and virtue != "rationale":
                virtue_value = avg_virtues[virtue]
                # Higher virtue values get higher alignment scores
                alignment = virtue_value * weight
                total_score += alignment
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_context_bonus(self, game_name: str, query_context: str) -> float:
        """Calculate context-based bonus for game selection"""
        context_lower = query_context.lower()

        # Context keywords that favor specific games
        context_bonuses = {
            "prisoner": ["cooperation", "defection", "dilemma", "trust", "fairness"],
            "auction": ["competition", "bidding", "resources", "allocation", "strategy"],
            "stackelberg": ["leadership", "hierarchy", "follower", "dominance"],
            "evolution": ["adaptation", "survival", "change", "evolution", "progress"],
            "regret_min": ["learning", "mistakes", "optimization", "improvement"],
            "shadow_play": ["unconscious", "shadow", "integration", "self-awareness"],
            "multiplay": ["collective", "group", "society", "coordination", "team"]
        }

        bonus = 0.0
        if game_name in context_bonuses:
            keywords = context_bonuses[game_name]
            matches = sum(1 for keyword in keywords if keyword in context_lower)
            bonus = min(matches * 0.1, 0.3)  # Max 0.3 bonus

        return bonus

    def get_sequential_plan(self, selected_games: List[GameRecommendation]) -> List[Dict[str, Any]]:
        """
        Create execution plan for sequential game application
        Returns list of game execution steps with dependencies
        """
        plan = []

        for i, game_rec in enumerate(selected_games):
            step = {
                "step": i + 1,
                "game": game_rec.game_name,
                "priority": game_rec.priority,
                "rationale": game_rec.rationale,
                "virtue_focus": game_rec.virtue_alignment,
                "depends_on": [g.game_name for g in selected_games[:i]],  # Previous games
                "nash_check_after": True,  # Check Nash equilibrium after each game
                "resource_allocation": self._calculate_resource_allocation(game_rec)
            }
            plan.append(step)

        return plan

    def _calculate_resource_allocation(self, game_rec: GameRecommendation) -> Dict[str, float]:
        """Calculate resource allocation to prevent potency overload"""
        base_allocation = {
            "cpu_percent": 0.1 + (game_rec.priority * 0.2),  # 10-30% CPU
            "memory_mb": 50 + (game_rec.priority * 100),     # 50-150MB
            "api_calls": 1 + int(game_rec.priority * 3),     # 1-4 API calls
            "processing_time_sec": 5 + (game_rec.priority * 10)  # 5-15 seconds
        }
        return base_allocation

# Global instance for easy access
game_selector = DynamicGameSelector()