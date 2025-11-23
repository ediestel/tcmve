#!/usr/bin/env python3
"""
ARCHER-1.0 Intelligence Enhancement Trial

Research Protocol: bIQ vs eIQ Analysis Across Gardner's Multiple Intelligences

Trial Design:
- Random sample of simulated persons across bIQ spectrum (Gaussian distribution)
- 7 Gardner intelligences tested: Linguistic, Logical-Mathematical, Spatial,
  Bodily-Kinesthetic, Musical, Interpersonal, Intrapersonal
- Compare biological IQ (bIQ) vs effective IQ (eIQ) with LLM enhancement
- Analyze enhancement patterns: linear, diminishing returns, equal gains, etc.

TCMVE Integration:
- Uses Thomistic virtue calculus for intelligence enhancement modeling
- eIQ = bIQ + enhancement_factor * virtue_coefficient
- Tracks metaphysical analysis of intelligence amplification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from backend.tcmve import TCMVE
from types import SimpleNamespace

# Gardner's Multiple Intelligences
GARDNER_INTELLIGENCES = [
    "Linguistic",
    "Logical_Mathematical",
    "Spatial",
    "Bodily_Kinesthetic",
    "Musical",
    "Interpersonal",
    "Intrapersonal"
]

@dataclass
class SimulatedPerson:
    """Represents a simulated person in the trial"""
    person_id: int
    biq: float  # Biological IQ
    virtues: Dict[str, float]  # Thomistic virtues
    gardner_scores: Dict[str, float]  # Base Gardner intelligence scores

@dataclass
class TrialResult:
    """Results for one intelligence test"""
    person_id: int
    intelligence_type: str
    biq_score: float
    eiq_score: float
    enhancement_ratio: float
    virtue_coefficient: float

class ARCHERTrial:
    """ARCHER-1.0 Intelligence Enhancement Research Trial"""

    def __init__(self,
                 num_persons: int = 240,
                 mean_biq: float = 100,
                 sigma_biq: float = 15,
                 llm_provider: str = "openai"):
        self.num_persons = num_persons
        self.mean_biq = mean_biq
        self.sigma_biq = sigma_biq
        self.llm_provider = llm_provider

        # Initialize TCMVE engine for enhancement calculations
        # Create basic args object for TCMVE
        basic_args = SimpleNamespace(
            use_generator=True,
            use_verifier=True,
            use_arbiter=True,
            generator_provider="openai",
            verifier_provider="openai",
            arbiter_provider="openai",
            max_rounds=5,
            marital_freedom=False,
            vice_check=True,
            self_refine=False,
            stream_mode=False,
            game_mode="all",
            selected_game=None,
            eiq_level=10,
            simulated_persons=240,
            mean_biq=100,
            sigma_biq=15,
            biq_distribution="gaussian",
            virtues_independent=False,
            output="archer_analysis",
            nash_mode="auto",
            tlpo_full=False,
            no_xml=False,
            seven_domains=False
        )
        self.tcmve = TCMVE(max_rounds=5, args=basic_args)

        # Trial data
        self.persons: List[SimulatedPerson] = []
        self.results: List[TrialResult] = []

    def generate_population(self) -> None:
        """Generate random sample of persons with Gaussian bIQ distribution"""
        print(f"Generating population of {self.num_persons} simulated persons...")

        # Generate bIQ scores (Gaussian distribution)
        biq_scores = np.random.normal(self.mean_biq, self.sigma_biq, self.num_persons)

        for i, biq in enumerate(biq_scores):
            # Generate virtue vectors (correlated with bIQ for realism)
            virtues = self._generate_virtues(biq)

            # Generate Gardner intelligence scores (correlated with bIQ)
            gardner_scores = self._generate_gardner_scores(biq)

            person = SimulatedPerson(
                person_id=i+1,
                biq=float(biq),
                virtues=virtues,
                gardner_scores=gardner_scores
            )
            self.persons.append(person)

        print(f"Population generated. bIQ range: {biq_scores.min():.1f} - {biq_scores.max():.1f}")

    def _generate_virtues(self, biq: float) -> Dict[str, float]:
        """Generate Thomistic virtue vector correlated with bIQ"""
        # Higher bIQ correlates with higher virtues (intelligence as virtue manifestation)
        base_level = 0.7 + (biq - 70) / 200  # 70-130 IQ maps to 0.7-0.95 virtue range
        base_level = np.clip(base_level, 0.5, 0.98)

        # Add some random variation
        virtues = {}
        for virtue in ['P', 'J', 'F', 'T', 'V', 'L', 'H', 'Ω']:
            variation = np.random.normal(0, 0.1)
            virtues[virtue] = np.clip(base_level + variation, 0.1, 1.0)

        return virtues

    def _generate_gardner_scores(self, biq: float) -> Dict[str, float]:
        """Generate Gardner intelligence scores correlated with bIQ"""
        scores = {}
        for intelligence in GARDNER_INTELLIGENCES:
            # Base score correlated with bIQ, with some domain-specific variation
            base_score = biq + np.random.normal(0, 10)
            # Some intelligences have different correlations
            if intelligence in ["Logical_Mathematical", "Spatial"]:
                base_score += np.random.normal(5, 3)  # Academic intelligences slightly higher
            elif intelligence in ["Bodily_Kinesthetic", "Musical"]:
                base_score += np.random.normal(-2, 5)  # Practical intelligences more variable

            scores[intelligence] = np.clip(base_score, 40, 160)

        return scores

    def calculate_eiq_enhancement(self, person: SimulatedPerson, intelligence_type: str) -> Tuple[float, float]:
        """
        Calculate eIQ enhancement for a specific intelligence using TCMVE virtue calculus

        CRUCIAL CLARIFICATION: eIQ allows using LLM to SOLVE Gardner's tests themselves.
        This is not iterative self-improvement, but immediate augmentation where the LLM
        serves as a cognitive extension. Enhancement should be MASSIVE even with 1 round.

        The model now reflects:
        - bIQ: Biological intelligence solving tests unaided
        - eIQ: Biological intelligence + LLM assistance solving the same tests
        - Enhancement represents the multiplicative boost from cognitive augmentation

        Returns: (eiq_score, enhancement_ratio)
        """
        biq_score = person.gardner_scores[intelligence_type]
        biq_percentile = stats.norm.cdf((person.biq - self.mean_biq) / self.sigma_biq)  # bIQ percentile

        # Use TCMVE virtue calculus for enhancement modeling
        virtues = person.virtues
        P, J, F, T, V, L, H, Ω = virtues['P'], virtues['J'], virtues['F'], virtues['T'], virtues['V'], virtues['L'], virtues['H'], virtues['Ω']
        virtue_coefficient = (P * J * F * T * V * L * H * Ω) / 1000

        # Domain-specific LLM augmentation factors (representing capability boost in each domain)
        # These represent the DIRECT IQ boost when LLM assists with solving tests
        # Based on user's examples: language mastery, PhD problem solving, code writing, etc.
        domain_augmentation = {
            "Linguistic": 50.0,     # Language tasks: near-universal communication, translation
            "Logical_Mathematical": 80.0,  # Math tasks: PhD-level problem solving, proofs
            "Spatial": 40.0,        # Spatial tasks: advanced visualization, design
            "Bodily_Kinesthetic": 10.0,    # Physical tasks: limited motor skill assistance
            "Musical": 25.0,        # Musical tasks: composition, analysis, theory
            "Interpersonal": 45.0,  # Social tasks: deep empathy, communication
            "Intrapersonal": 30.0   # Self-awareness: profound introspection
        }

        base_augmentation = domain_augmentation.get(intelligence_type, 30.0)

        # eIQ MODEL: LLM-assisted test performance
        # The virtue coefficient modulates HOW WELL the person leverages LLM capabilities
        # But the base LLM capability is substantial regardless of virtue alignment

        # Core augmentation: LLM provides direct capability boost
        # Virtue coefficient affects utilization efficiency (0.1 to 1.0 range)
        virtue_efficiency = min(1.0, max(0.1, virtue_coefficient * 1000))  # Scale virtue coeff to 0.1-1.0
        llm_boost = base_augmentation * virtue_efficiency

        # IQ group differential: How effectively different groups leverage LLM assistance
        # Hypothesis testing: Does bIQ level affect LLM utilization effectiveness?

        # Model 1: Equal augmentation for all (pure democratizing effect)
        equal_model = llm_boost

        # Model 2: Higher bIQ leverages LLM better (rich get richer)
        rich_richer_model = llm_boost * (1 + 0.5 * biq_percentile)

        # Model 3: Lower bIQ benefits more from LLM assistance (leveling effect)
        leveling_model = llm_boost * (1 + 0.5 * (1 - biq_percentile))

        # Model 4: Optimal leverage at middle IQ levels (Goldilocks effect)
        goldilocks_model = llm_boost * (1 + 0.3 * np.exp(-2 * (biq_percentile - 0.5)**2))

        # SELECT MODEL: Using rich-get-richer as default hypothesis to test
        selected_enhancement = rich_richer_model

        # Add domain-specific and individual variation
        domain_noise = np.random.normal(1.0, 0.1)  # 10% variation
        individual_noise = np.random.normal(1.0, 0.05)  # 5% individual variation

        total_enhancement = selected_enhancement * domain_noise * individual_noise

        # Calculate final eIQ: biological + LLM-augmented performance
        eiq_score = biq_score + total_enhancement

        # eIQ can significantly exceed bIQ due to LLM augmentation
        # No upper limit - reflects potential for superhuman performance
        eiq_score = max(eiq_score, biq_score)  # Minimum is original performance

        enhancement_ratio = eiq_score / biq_score

        return eiq_score, enhancement_ratio

    def run_trial(self) -> None:
        """Execute the full ARCHER-1.0 trial"""
        print("🏹 ARCHER-1.0 INTELLIGENCE ENHANCEMENT TRIAL 🏹")
        print("=" * 60)
        print(f"Sample Size: {self.num_persons} simulated persons")
        print(f"bIQ Distribution: Gaussian(μ={self.mean_biq}, σ={self.sigma_biq})")
        print(f"LLM Provider: {self.llm_provider}")
        print("=" * 60)

        if not self.persons:
            self.generate_population()

        print("\nRunning intelligence assessments...")

        for person in self.persons:
            for intelligence in GARDNER_INTELLIGENCES:
                eiq_score, enhancement_ratio = self.calculate_eiq_enhancement(person, intelligence)

                result = TrialResult(
                    person_id=person.person_id,
                    intelligence_type=intelligence,
                    biq_score=person.gardner_scores[intelligence],
                    eiq_score=eiq_score,
                    enhancement_ratio=enhancement_ratio,
                    virtue_coefficient=(person.virtues['P'] * person.virtues['J'] * person.virtues['F'] *
                                      person.virtues['T'] * person.virtues['V'] * person.virtues['L'] *
                                      person.virtues['H'] * person.virtues['Ω']) / 1000
                )
                self.results.append(result)

            # Progress indicator
            if person.person_id % 50 == 0:
                print(f"Processed {person.person_id}/{self.num_persons} persons...")

        print(f"\n✅ Trial complete! Collected {len(self.results)} intelligence assessments")

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze trial results for patterns and insights"""
        print("\n🔍 ANALYZING RESULTS...")
        print("=" * 60)

        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'person_id': r.person_id,
            'intelligence': r.intelligence_type,
            'biq': r.biq_score,
            'eiq': r.eiq_score,
            'enhancement_ratio': r.enhancement_ratio,
            'virtue_coeff': r.virtue_coefficient
        } for r in self.results])

        analysis = {}

        # Overall enhancement statistics
        overall_enhancement = df['enhancement_ratio'].mean()
        print(".2f")
        print(".1f")
        print(".1f")

        # Pattern analysis
        patterns = self._analyze_enhancement_patterns(df)
        analysis['patterns'] = patterns

        # Intelligence-specific analysis
        intelligence_analysis = self._analyze_by_intelligence(df)
        analysis['by_intelligence'] = intelligence_analysis

        # bIQ correlation analysis (now with multiplicative enhancement)
        biq_correlation = self._analyze_biq_correlation(df)
        analysis['biq_correlation'] = biq_correlation

        # Virtue coefficient analysis
        virtue_analysis = self._analyze_virtue_impact(df)
        analysis['virtue_analysis'] = virtue_analysis

        # IQ group response analysis (key research question)
        iq_group_analysis = self._analyze_iq_group_responses(df)
        analysis['iq_group_responses'] = iq_group_analysis

        # Generate visualizations
        self._generate_visualizations(df)

        return analysis

    def _analyze_enhancement_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze different enhancement patterns"""
        patterns = {}

        # Linear enhancement test
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['biq'], df['enhancement_ratio'])
        patterns['linear_enhancement'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        # Diminishing returns test (quadratic fit)
        coeffs = np.polyfit(df['biq'], df['enhancement_ratio'], 2)
        patterns['diminishing_returns'] = {
            'quadratic_coeff': coeffs[0],
            'linear_coeff': coeffs[1],
            'constant': coeffs[2],
            'is_diminishing': coeffs[0] < 0  # Negative quadratic coefficient
        }

        # Equal gains test (ANOVA)
        groups = [df[df['biq'].between(i, i+10)]['enhancement_ratio'] for i in range(70, 131, 10)]
        f_stat, p_value = stats.f_oneway(*groups)
        patterns['equal_gains'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'equal_gains_supported': p_value > 0.05  # No significant difference between groups
        }

        # Threshold effects (piecewise regression approximation)
        low_biq = df[df['biq'] < 100]['enhancement_ratio'].mean()
        high_biq = df[df['biq'] >= 100]['enhancement_ratio'].mean()
        patterns['threshold_effects'] = {
            'low_biq_avg_enhancement': low_biq,
            'high_biq_avg_enhancement': high_biq,
            'threshold_at_100': high_biq > low_biq,
            'enhancement_ratio': high_biq / low_biq if low_biq > 0 else float('inf'),
            'absolute_gap': high_biq - low_biq
        }

        # Ceiling effects
        max_eiq = df['eiq'].max()
        ceiling_reached = df[df['eiq'] >= 190]['enhancement_ratio'].mean()
        patterns['ceiling_effects'] = {
            'max_eiq_observed': max_eiq,
            'ceiling_enhancement_avg': ceiling_reached,
            'ceiling_reached': max_eiq >= 195
        }

        print("Pattern Analysis Results:")
        for pattern_name, results in patterns.items():
            print(f"  {pattern_name}: {results}")

        return patterns

    def _analyze_by_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze enhancement by intelligence type"""
        analysis = {}

        for intelligence in GARDNER_INTELLIGENCES:
            subset = df[df['intelligence'] == intelligence]
            analysis[intelligence] = {
                'avg_biq': subset['biq'].mean(),
                'avg_eiq': subset['eiq'].mean(),
                'avg_enhancement': subset['enhancement_ratio'].mean(),
                'enhancement_std': subset['enhancement_ratio'].std(),
                'sample_size': len(subset)
            }

        # Sort by enhancement ratio
        sorted_intelligences = sorted(analysis.items(), key=lambda x: x[1]['avg_enhancement'], reverse=True)

        print("\nIntelligence-Specific Enhancement (eIQ/bIQ ratios):")
        for intel, stats in sorted_intelligences:
            print(".1f")

        return analysis

    def _analyze_biq_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation between bIQ and enhancement"""
        correlation = stats.pearsonr(df['biq'], df['enhancement_ratio'])

        # Binned analysis
        bins = pd.cut(df['biq'], bins=5)
        binned_means = df.groupby(bins)['enhancement_ratio'].mean()

        analysis = {
            'overall_correlation': correlation[0],
            'correlation_p_value': correlation[1],
            'binned_means': binned_means.to_dict(),
            'correlation_strength': self._interpret_correlation(correlation[0])
        }

        print("\nbIQ-Enhancement Correlation:")
        print(f"  Correlation coefficient: {correlation[0]:.3f}")
        print(f"  Correlation strength: {analysis['correlation_strength']}")
        print(".1f")
        print(".1f")

        return analysis

    def _analyze_virtue_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how Thomistic virtues impact enhancement"""
        correlation = stats.pearsonr(df['virtue_coeff'], df['enhancement_ratio'])

        analysis = {
            'virtue_enhancement_correlation': correlation[0],
            'correlation_p_value': correlation[1],
            'virtue_coefficient_range': [df['virtue_coeff'].min(), df['virtue_coeff'].max()],
            'avg_virtue_coefficient': df['virtue_coeff'].mean()
        }

        print("\nThomistic Virtue Analysis:")
        print(f"  Virtue-enhancement correlation: {correlation[0]:.3f}")
        print(f"  Virtue coefficient range: {df['virtue_coeff'].min():.4f} - {df['virtue_coeff'].max():.4f}")

        return analysis

    def _analyze_iq_group_responses(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how different IQ groups respond to LLM enhancement (key research question)"""
        print("\n🧠 IQ GROUP RESPONSE ANALYSIS (Key Research Question)")
        print("=" * 60)

        # Define IQ groups based on standard deviations from mean
        iq_groups = {
            'Very Low (<-2σ)': df[df['biq'] < self.mean_biq - 2*self.sigma_biq],
            'Low (-2σ to -1σ)': df[(df['biq'] >= self.mean_biq - 2*self.sigma_biq) & (df['biq'] < self.mean_biq - self.sigma_biq)],
            'Average (-1σ to +1σ)': df[(df['biq'] >= self.mean_biq - self.sigma_biq) & (df['biq'] <= self.mean_biq + self.sigma_biq)],
            'High (+1σ to +2σ)': df[(df['biq'] > self.mean_biq + self.sigma_biq) & (df['biq'] <= self.mean_biq + 2*self.sigma_biq)],
            'Very High (>+2σ)': df[df['biq'] > self.mean_biq + 2*self.sigma_biq]
        }

        analysis = {}

        print("Enhancement by IQ Group:")
        print("Group\t\tSample Size\tAvg bIQ\tAvg eIQ\tAvg Enhancement\tStd Enhancement")
        print("-" * 90)

        for group_name, group_data in iq_groups.items():
            if len(group_data) == 0:
                continue

            avg_biq = group_data['biq'].mean()
            avg_eiq = group_data['eiq'].mean()
            avg_enhancement = group_data['enhancement_ratio'].mean()
            std_enhancement = group_data['enhancement_ratio'].std()

            analysis[group_name] = {
                'sample_size': len(group_data),
                'avg_biq': avg_biq,
                'avg_eiq': avg_eiq,
                'avg_enhancement_ratio': avg_enhancement,
                'std_enhancement_ratio': std_enhancement,
                'enhancement_range': [group_data['enhancement_ratio'].min(), group_data['enhancement_ratio'].max()]
            }

            print(".1f")

        # Test hypotheses about group differences
        hypotheses = self._test_enhancement_hypotheses(iq_groups)
        analysis['hypotheses_tests'] = hypotheses

        # Analyze consistency across intelligence domains
        domain_consistency = self._analyze_domain_consistency(df, iq_groups)
        analysis['domain_consistency'] = domain_consistency

        return analysis

    def _test_enhancement_hypotheses(self, iq_groups: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Test different hypotheses about how IQ groups respond to enhancement"""
        # Extract enhancement ratios for each group (non-empty groups only)
        group_enhancements = {}
        for name, data in iq_groups.items():
            if len(data) > 0:
                group_enhancements[name] = data['enhancement_ratio'].values

        hypotheses = {}

        # Hypothesis 1: Equal enhancement across all groups (ANOVA test)
        if len(group_enhancements) > 1:
            try:
                f_stat, p_value = stats.f_oneway(*group_enhancements.values())
                hypotheses['equal_enhancement'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'equal_enhancement_supported': p_value > 0.05,
                    'interpretation': 'Groups receive equal enhancement' if p_value > 0.05 else 'Groups receive different enhancement levels'
                }
            except:
                hypotheses['equal_enhancement'] = {'error': 'Could not perform ANOVA'}

        # Hypothesis 2: Rich get richer (correlation between bIQ and enhancement)
        all_biq = []
        all_enhancement = []
        for name, data in iq_groups.items():
            all_biq.extend(data['biq'].tolist())
            all_enhancement.extend(data['enhancement_ratio'].tolist())

        if all_biq and all_enhancement:
            correlation = stats.pearsonr(all_biq, all_enhancement)
            hypotheses['rich_get_richer'] = {
                'correlation': correlation[0],
                'p_value': correlation[1],
                'supported': correlation[0] > 0.3 and correlation[1] < 0.05,
                'interpretation': 'Higher bIQ correlates with greater enhancement' if correlation[0] > 0.3 else 'No rich-get-richer effect'
            }

        # Hypothesis 3: Leveling effect (lower bIQ gets more relative enhancement)
        # Compare lowest and highest groups
        group_names = list(group_enhancements.keys())
        if len(group_names) >= 2:
            lowest_group = group_names[0]  # Assuming ordered from lowest to highest
            highest_group = group_names[-1]

            lowest_avg = np.mean(group_enhancements[lowest_group])
            highest_avg = np.mean(group_enhancements[highest_group])

            hypotheses['leveling_effect'] = {
                'lowest_group_avg': lowest_avg,
                'highest_group_avg': highest_avg,
                'ratio': lowest_avg / highest_avg,
                'supported': lowest_avg > highest_avg,
                'interpretation': 'Lower IQ groups benefit more (leveling)' if lowest_avg > highest_avg else 'No leveling effect observed'
            }

        print("\nHypothesis Testing Results:")
        for hyp_name, results in hypotheses.items():
            if 'interpretation' in results:
                print(f"  {hyp_name}: {results['interpretation']}")

        return hypotheses

    def _analyze_domain_consistency(self, df: pd.DataFrame, iq_groups: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze if enhancement patterns are consistent across intelligence domains"""
        domain_consistency = {}

        for intelligence in GARDNER_INTELLIGENCES:
            domain_data = df[df['intelligence'] == intelligence]
            domain_groups = {}

            # Split domain data by IQ groups
            for group_name, group_data in iq_groups.items():
                domain_group_data = domain_data[domain_data['biq'].isin(group_data['biq'])]
                if len(domain_group_data) > 0:
                    domain_groups[group_name] = domain_group_data['enhancement_ratio'].mean()

            domain_consistency[intelligence] = domain_groups

        # Calculate consistency metrics
        consistency_analysis = {}
        for intelligence, group_means in domain_consistency.items():
            if len(group_means) > 1:
                mean_enhancement = np.mean(list(group_means.values()))
                std_enhancement = np.std(list(group_means.values()))
                cv = std_enhancement / mean_enhancement if mean_enhancement > 0 else 0
                consistency_analysis[intelligence] = {
                    'coefficient_of_variation': cv,
                    'consistency': 'High' if cv < 0.2 else 'Moderate' if cv < 0.4 else 'Low'
                }

        print("\nDomain Consistency Analysis:")
        print("Intelligence\t\tConsistency\tCV")
        print("-" * 40)
        for intel, metrics in consistency_analysis.items():
            print(f"{intel[:15]}\t\t{metrics['consistency']}\t\t{metrics['coefficient_of_variation']:.3f}")

        return {
            'by_domain': domain_consistency,
            'consistency_metrics': consistency_analysis
        }

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient strength"""
        abs_r = abs(r)
        if abs_r < 0.3:
            return "weak"
        elif abs_r < 0.5:
            return "moderate"
        elif abs_r < 0.7:
            return "strong"
        else:
            return "very strong"

    def _generate_visualizations(self, df: pd.DataFrame) -> None:
        """Generate analysis visualizations"""
        print("\n📊 Generating visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. Enhancement vs bIQ scatter plot
        axes[0,0].scatter(df['biq'], df['enhancement_ratio'], alpha=0.6, s=10)
        axes[0,0].set_xlabel('Biological IQ (bIQ)')
        axes[0,0].set_ylabel('Enhancement Ratio (eIQ/bIQ)')
        axes[0,0].set_title('Intelligence Enhancement vs Biological IQ')
        axes[0,0].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(df['biq'], df['enhancement_ratio'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(df['biq'].min(), df['biq'].max(), 100)
        axes[0,0].plot(x_trend, p(x_trend), "r--", alpha=0.8, label='Quadratic trend')
        axes[0,0].legend()

        # 2. Enhancement by intelligence type
        intelligence_means = df.groupby('intelligence')['enhancement_ratio'].mean().sort_values(ascending=True)
        intelligence_means.plot(kind='barh', ax=axes[0,1])
        axes[0,1].set_xlabel('Average Enhancement Ratio')
        axes[0,1].set_title('Enhancement by Gardner Intelligence Type')
        axes[0,1].grid(True, alpha=0.3)

        # 3. bIQ distribution
        axes[1,0].hist(df['biq'], bins=30, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Biological IQ (bIQ)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('bIQ Distribution in Trial Population')
        axes[1,0].axvline(df['biq'].mean(), color='red', linestyle='--', label='.1f')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 4. Enhancement distribution (log scale for large ranges)
        axes[1,1].hist(df['enhancement_ratio'], bins=np.logspace(np.log10(df['enhancement_ratio'].min()),
                                                                np.log10(df['enhancement_ratio'].max()), 30),
                      alpha=0.7, edgecolor='black', color='green')
        axes[1,1].set_xlabel('Enhancement Ratio (eIQ/bIQ)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Enhancement Ratio Distribution (Log Scale)')
        axes[1,1].set_xscale('log')
        axes[1,1].axvline(df['enhancement_ratio'].mean(), color='red', linestyle='--',
                         label='.1f')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # 5. IQ Group Enhancement Analysis (Key Research Question)
        iq_groups = ['Very Low', 'Low', 'Average', 'High', 'Very High']
        iq_group_data = df.copy()
        iq_group_data['iq_group'] = pd.cut(iq_group_data['biq'],
                                         bins=[0, self.mean_biq-2*self.sigma_biq, self.mean_biq-self.sigma_biq,
                                               self.mean_biq+self.sigma_biq, self.mean_biq+2*self.sigma_biq, 200],
                                         labels=iq_groups)

        group_means = iq_group_data.groupby('iq_group')['enhancement_ratio'].mean()
        group_means.plot(kind='bar', ax=axes[1,2], color='purple', alpha=0.7)
        axes[1,2].set_xlabel('IQ Group')
        axes[1,2].set_ylabel('Average Enhancement Ratio')
        axes[1,2].set_title('Enhancement by IQ Group\n(Key Research Question)')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(group_means):
            axes[1,2].text(i, v + 0.01, '.1f', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('archer_trial_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved to 'archer_trial_analysis.png'")

    def save_results(self, filename: str = "archer_trial_results.json") -> None:
        """Save trial results to JSON file in generic format"""
        # Calculate analysis data
        df = pd.DataFrame([{
            'biq': r.biq_score,
            'enhancement': r.enhancement_ratio
        } for r in self.results])

        average_enhancement = df['enhancement'].mean()
        biq_enhancement_correlation = stats.pearsonr(df['biq'], df['enhancement'])[0]

        # IQ group analysis
        iq_groups = {}
        for i in range(70, 141, 10):
            group_data = df[df['biq'].between(i, i+9)]
            if len(group_data) > 0:
                iq_groups[f"{i}-{i+9}"] = {
                    'count': len(group_data),
                    'avg_biq': group_data['biq'].mean(),
                    'avg_eiq': (group_data['biq'] * (1 + group_data['enhancement'])).mean(),
                    'avg_enhancement': group_data['enhancement'].mean()
                }

        # Gardner analysis
        gardner_analysis = {}
        for intelligence in GARDNER_INTELLIGENCES:
            intel_results = [r for r in self.results if r.intelligence_type == intelligence]
            if intel_results:
                enhancements = [r.enhancement_ratio for r in intel_results]
                biq_scores = [r.biq_score for r in intel_results]
                gardner_analysis[intelligence] = {
                    'avg_enhancement': np.mean(enhancements),
                    'correlation': stats.pearsonr(biq_scores, enhancements)[0] if len(biq_scores) > 1 else 0
                }

        results_data = {
            'trial_type': 'ARCHER-1.0',
            'metadata': {
                'trial_name': 'ARCHER-1.0 Intelligence Enhancement Trial',
                'timestamp': pd.Timestamp.now().isoformat(),
                'num_persons': self.num_persons,
                'mean_biq': self.mean_biq,
                'sigma_biq': self.sigma_biq,
                'llm_provider': self.llm_provider,
                'version': 'ARCHER-1.0'
            },
            'analysis': {
                'average_enhancement': average_enhancement,
                'biq_enhancement_correlation': biq_enhancement_correlation,
                'enhancement_patterns': {
                    'linear': abs(biq_enhancement_correlation) > 0.3,
                    'diminishing_returns': self._check_diminishing_returns(df),
                    'equal_gains': self._check_equal_gains(df),
                    'threshold_effects': self._check_threshold_effects(df),
                    'ceiling_effects': df['enhancement'].max() > 1.8
                },
                'gardner_analysis': gardner_analysis,
                'iq_group_analysis': iq_groups
            },
            'persons': [{
                'person_id': p.person_id,
                'biq': p.biq,
                'virtues': p.virtues,
                'gardner_scores': p.gardner_scores
            } for p in self.persons],
            'results': [{
                'person_id': r.person_id,
                'intelligence_type': r.intelligence_type,
                'biq_score': r.biq_score,
                'eiq_score': r.eiq_score,
                'enhancement_ratio': r.enhancement_ratio,
                'virtue_coefficient': r.virtue_coefficient
            } for r in self.results]
        }

        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"✅ Results saved to '{filepath}'")

    def run_metaphysical_analysis(self) -> str:
        """Run TCMVE metaphysical analysis of the trial results"""
        print("\n🏛️  TCMVE METAPHYSICAL ANALYSIS 🏛️")

        # Summarize key findings
        df = pd.DataFrame([{
            'biq': r.biq_score,
            'enhancement': r.enhancement_ratio
        } for r in self.results])

        avg_enhancement = df['enhancement'].mean()
        correlation = stats.pearsonr(df['biq'], df['enhancement'])[0]

        # Create analysis query for TCMVE
        analysis_query = f"""
        Analyze the ARCHER-1.0 intelligence enhancement trial results from Thomistic perspective:

        Trial Results Summary:
        - Average enhancement ratio: {avg_enhancement:.3f}
        - bIQ-enhancement correlation: {correlation:.3f}
        - Sample size: {self.num_persons} persons across 7 Gardner intelligences

        Key Patterns Observed:
        1. Linear Enhancement: {correlation > 0.3}
        2. Diminishing Returns: {self._check_diminishing_returns(df)}
        3. Equal Gains: {self._check_equal_gains(df)}
        4. Threshold Effects: {self._check_threshold_effects(df)}
        5. Ceiling Effects: {df['enhancement'].max() > 1.8}

        Provide Four Causes analysis of intelligence amplification through LLM enhancement.
        What is the formal cause (essence) of enhanced intelligence?
        What is the final cause (telos/purpose) of this technological augmentation?
        """

        print("Running TCMVE analysis of trial results...")
        result = self.tcmve.run(analysis_query, self.tcmve.args)

        return result['final_answer']

    def _check_diminishing_returns(self, df: pd.DataFrame) -> bool:
        """Check if diminishing returns pattern is present"""
        coeffs = np.polyfit(df['biq'], df['enhancement'], 2)
        return coeffs[0] < -0.0001  # Negative quadratic coefficient

    def _check_equal_gains(self, df: pd.DataFrame) -> bool:
        """Check if equal gains pattern is present"""
        groups = [df[df['biq'].between(i, i+10)]['enhancement'].mean()
                 for i in range(70, 131, 10)]
        return np.std(groups) < 0.05  # Low variance between groups

    def _check_threshold_effects(self, df: pd.DataFrame) -> bool:
        """Check if threshold effects are present"""
        low_group = df[df['biq'] < 100]['enhancement'].mean()
        high_group = df[df['biq'] >= 100]['enhancement'].mean()
        return abs(high_group - low_group) > 0.1  # Significant difference

def main():
    """Run the ARCHER-1.0 trial"""
    print("🏹 ARCHER-1.0 INTELLIGENCE ENHANCEMENT RESEARCH TRIAL 🏹")
    print("Thomistic Cross-Model Verification Engine")
    print("=" * 80)

    # Initialize trial
    trial = ARCHERTrial(
        num_persons=240,  # Standard sample size
        mean_biq=100,     # Population mean IQ
        sigma_biq=15,     # Population IQ standard deviation
        llm_provider="openai"
    )

    # Run trial
    trial.run_trial()

    # Analyze results
    analysis = trial.analyze_results()

    # Run metaphysical analysis
    metaphysical_analysis = trial.run_metaphysical_analysis()

    # Save results
    trial.save_results()

    # Print final report
    print("\n" + "=" * 80)
    print("🏹 ARCHER-1.0 TRIAL COMPLETE 🏹")
    print("=" * 80)
    print("\nTCMVE METAPHYSICAL ANALYSIS:")
    print("-" * 40)
    print(metaphysical_analysis)
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
