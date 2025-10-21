"""
OpenCog-inspired cognitive adaptation system for customer service agents.

This module implements core cognitive concepts from OpenCog:
- AtomSpace-like knowledge representation
- Cognitive attention allocation
- Adaptive learning from interactions
- Memory formation and retrieval
"""

from __future__ import annotations

import json
import time
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math

class AtomType(Enum):
    """Types of atoms in our cognitive atomspace."""
    CONCEPT = "ConceptNode"
    PREDICATE = "PredicateNode"  
    LINK = "Link"
    EVALUATION = "EvaluationLink"
    INHERITANCE = "InheritanceLink"
    SIMILARITY = "SimilarityLink"
    CONTEXT = "ContextLink"
    PATTERN = "PatternNode"

@dataclass
class TruthValue:
    """Truth value representing strength and confidence of beliefs."""
    strength: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.1  # 0.0 to 1.0
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))

@dataclass 
class AttentionValue:
    """Attention value for cognitive resource allocation."""
    sti: float = 0.0  # Short-term importance
    lti: float = 0.0  # Long-term importance  
    vlti: float = 0.0  # Very long-term importance
    
    def decay(self, factor: float = 0.95):
        """Apply attention decay over time."""
        self.sti *= factor
        self.lti *= factor * 0.99
        self.vlti *= factor * 0.999

@dataclass
class Atom:
    """Basic unit of knowledge in the cognitive system."""
    name: str
    atom_type: AtomType
    truth_value: TruthValue = field(default_factory=TruthValue)
    attention_value: AttentionValue = field(default_factory=AttentionValue)
    incoming: Set[str] = field(default_factory=set)
    outgoing: List[str] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash(self.name)

class CognitiveAtomSpace:
    """AtomSpace-like knowledge representation system."""
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}
        self.attention_bank = AttentionalBank()
        self.learning_events = deque(maxlen=1000)
        
    def add_atom(self, name: str, atom_type: AtomType, 
                 truth_value: Optional[TruthValue] = None,
                 attention_value: Optional[AttentionValue] = None) -> Atom:
        """Add or retrieve an atom from the atomspace."""
        if name in self.atoms:
            atom = self.atoms[name]
            atom.last_accessed = time.time()
            return atom
            
        atom = Atom(
            name=name,
            atom_type=atom_type,
            truth_value=truth_value or TruthValue(),
            attention_value=attention_value or AttentionValue()
        )
        self.atoms[name] = atom
        return atom
    
    def add_link(self, link_name: str, link_type: AtomType, 
                 source: str, target: str,
                 truth_value: Optional[TruthValue] = None) -> Atom:
        """Create a link between atoms."""
        link = self.add_atom(link_name, link_type, truth_value)
        link.outgoing = [source, target]
        
        # Update incoming links
        if source in self.atoms:
            self.atoms[source].incoming.add(link_name)
        if target in self.atoms:
            self.atoms[target].incoming.add(link_name)
            
        return link
    
    def get_related_atoms(self, atom_name: str, max_depth: int = 2) -> Set[str]:
        """Get atoms related to the given atom within max_depth."""
        if atom_name not in self.atoms:
            return set()
            
        related = {atom_name}
        current_level = {atom_name}
        
        for _ in range(max_depth):
            next_level = set()
            for atom_name in current_level:
                atom = self.atoms[atom_name]
                # Add outgoing atoms
                next_level.update(atom.outgoing)
                # Add atoms that link to this one
                for link_name in atom.incoming:
                    if link_name in self.atoms:
                        link = self.atoms[link_name]
                        next_level.update(link.outgoing)
            
            current_level = next_level - related
            related.update(current_level)
            if not current_level:
                break
                
        return related
    
    def update_attention(self, atom_name: str, sti_boost: float = 10.0):
        """Boost attention for an atom and apply spreading activation."""
        if atom_name not in self.atoms:
            return
            
        atom = self.atoms[atom_name]
        atom.attention_value.sti += sti_boost
        atom.last_accessed = time.time()
        
        # Spread activation to related atoms
        related = self.get_related_atoms(atom_name, max_depth=1)
        for related_name in related:
            if related_name != atom_name and related_name in self.atoms:
                self.atoms[related_name].attention_value.sti += sti_boost * 0.3
    
    def decay_attention(self):
        """Apply attention decay to all atoms."""
        for atom in self.atoms.values():
            atom.attention_value.decay()
    
    def get_high_attention_atoms(self, limit: int = 10) -> List[Atom]:
        """Get atoms with highest attention values."""
        return sorted(
            self.atoms.values(),
            key=lambda a: a.attention_value.sti + a.attention_value.lti,
            reverse=True
        )[:limit]

class AttentionalBank:
    """Manages cognitive attention allocation."""
    
    def __init__(self):
        self.total_sti = 1000.0
        self.attention_allocation: Dict[str, float] = defaultdict(float)
        
    def allocate_attention(self, agent_name: str, amount: float) -> float:
        """Allocate attention to an agent, returns actual allocated amount."""
        available = max(0, self.total_sti - sum(self.attention_allocation.values()))
        allocated = min(amount, available)
        self.attention_allocation[agent_name] += allocated
        return allocated
    
    def release_attention(self, agent_name: str, amount: float):
        """Release attention from an agent back to the bank."""
        released = min(amount, self.attention_allocation[agent_name])
        self.attention_allocation[agent_name] -= released
        
    def get_attention_distribution(self) -> Dict[str, float]:
        """Get current attention distribution across agents."""
        return dict(self.attention_allocation)

class CognitiveMemory:
    """Episodic and semantic memory system."""
    
    def __init__(self):
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_patterns: Dict[str, float] = defaultdict(float)
        self.interaction_history: deque = deque(maxlen=100)
        
    def store_episode(self, context: Dict[str, Any], action: str, 
                     outcome: str, satisfaction: float):
        """Store an interaction episode."""
        episode = {
            'timestamp': time.time(),
            'context': context,
            'action': action,
            'outcome': outcome,
            'satisfaction': satisfaction,
            'context_hash': hash(json.dumps(context, sort_keys=True))
        }
        self.episodic_memory.append(episode)
        self.interaction_history.append(episode)
        
        # Update semantic patterns
        pattern_key = f"{action}_{outcome}"
        self.semantic_patterns[pattern_key] += satisfaction
    
    def retrieve_similar_episodes(self, context: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve episodes similar to current context."""
        context_hash = hash(json.dumps(context, sort_keys=True))
        
        # Simple similarity based on context overlap
        similar_episodes = []
        for episode in self.episodic_memory:
            similarity = self._calculate_similarity(context, episode['context'])
            if similarity > 0.3:  # Threshold for similarity
                episode_copy = episode.copy()
                episode_copy['similarity'] = similarity
                similar_episodes.append(episode_copy)
        
        # Sort by similarity and satisfaction
        similar_episodes.sort(
            key=lambda e: (e['similarity'] * e['satisfaction']), 
            reverse=True
        )
        return similar_episodes[:limit]
    
    def _calculate_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts."""
        keys1, keys2 = set(context1.keys()), set(context2.keys())
        common_keys = keys1 & keys2
        
        if not common_keys:
            return 0.0
            
        matches = sum(1 for k in common_keys if context1[k] == context2[k])
        return matches / len(keys1 | keys2)

class CognitiveAgentAdapter:
    """Cognitive adaptation layer for OpenAI agents."""
    
    def __init__(self):
        self.atomspace = CognitiveAtomSpace()
        self.memory = CognitiveMemory()
        self.agent_performance: Dict[str, List[float]] = defaultdict(list)
        self.context_agent_mapping: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
    def learn_from_interaction(self, agent_name: str, context: Dict[str, Any], 
                             user_message: str, agent_response: str, 
                             satisfaction: float = 0.5):
        """Learn from an agent interaction to improve future routing."""
        
        # Store in episodic memory
        self.memory.store_episode(context, agent_name, agent_response, satisfaction)
        
        # Update agent performance tracking
        self.agent_performance[agent_name].append(satisfaction)
        if len(self.agent_performance[agent_name]) > 20:
            self.agent_performance[agent_name].pop(0)
        
        # Update atomspace with knowledge
        self._update_atomspace_knowledge(agent_name, context, user_message, satisfaction)
        
        # Update context-agent mappings
        context_key = self._extract_context_features(context, user_message)
        self.context_agent_mapping[context_key][agent_name] += satisfaction * 0.1
        
    def suggest_best_agent(self, context: Dict[str, Any], user_message: str) -> Dict[str, float]:
        """Suggest the best agent based on cognitive analysis."""
        context_key = self._extract_context_features(context, user_message)
        
        # Get base suggestions from experience
        agent_scores = dict(self.context_agent_mapping[context_key])
        
        # Boost scores based on recent performance
        for agent_name, performances in self.agent_performance.items():
            if performances:
                recent_avg = sum(performances[-5:]) / len(performances[-5:])
                agent_scores[agent_name] = agent_scores.get(agent_name, 0) + recent_avg * 0.2
        
        # Apply attention-based boosting
        attention_dist = self.atomspace.attention_bank.get_attention_distribution()
        for agent_name, attention in attention_dist.items():
            normalized_attention = attention / 1000.0  # Normalize to 0-1
            agent_scores[agent_name] = agent_scores.get(agent_name, 0) + normalized_attention * 0.1
        
        # Retrieve similar episodes for additional context
        similar_episodes = self.memory.retrieve_similar_episodes(context)
        for episode in similar_episodes:
            action_agent = episode['action']
            boost = episode['satisfaction'] * episode['similarity'] * 0.15
            agent_scores[action_agent] = agent_scores.get(action_agent, 0) + boost
        
        return agent_scores
    
    def _extract_context_features(self, context: Dict[str, Any], user_message: str) -> str:
        """Extract key features from context and message for pattern matching."""
        features = []
        
        # Add context features
        for key, value in context.items():
            if value and key in ['passenger_name', 'confirmation_number', 'flight_number']:
                features.append(f"has_{key}")
        
        # Add message intent features
        message_lower = user_message.lower()
        intent_keywords = {
            'seat': ['seat', 'change seat', 'move'],
            'cancel': ['cancel', 'cancellation', 'refund'],
            'status': ['status', 'flight status', 'delayed', 'on time'],
            'faq': ['how many', 'what is', 'tell me about', 'baggage', 'wifi']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                features.append(f"intent_{intent}")
        
        return "_".join(sorted(features)) or "general"
    
    def _update_atomspace_knowledge(self, agent_name: str, context: Dict[str, Any], 
                                  user_message: str, satisfaction: float):
        """Update atomspace with interaction knowledge."""
        
        # Create atoms for key entities
        agent_atom = self.atomspace.add_atom(f"agent_{agent_name}", AtomType.CONCEPT)
        message_atom = self.atomspace.add_atom(f"message_intent_{self._extract_intent(user_message)}", AtomType.CONCEPT)
        
        # Update attention based on satisfaction
        attention_boost = satisfaction * 20.0
        self.atomspace.update_attention(agent_atom.name, attention_boost)
        self.atomspace.update_attention(message_atom.name, attention_boost)
        
        # Create evaluation link for agent-message satisfaction
        truth_value = TruthValue(strength=satisfaction, confidence=min(0.9, 0.1 + satisfaction * 0.8))
        self.atomspace.add_link(
            f"satisfaction_{agent_name}_{int(time.time())}",
            AtomType.EVALUATION,
            agent_atom.name,
            message_atom.name,
            truth_value
        )
        
        # Apply attention decay
        self.atomspace.decay_attention()
    
    def _extract_intent(self, message: str) -> str:
        """Extract intent from user message."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['seat', 'change', 'move']):
            return 'seat_change'
        elif any(word in message_lower for word in ['cancel', 'cancellation']):
            return 'cancellation'  
        elif any(word in message_lower for word in ['status', 'flight', 'delayed']):
            return 'flight_status'
        elif any(word in message_lower for word in ['baggage', 'bag', 'wifi', 'how many']):
            return 'faq'
        else:
            return 'general'
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state for visualization."""
        high_attention_atoms = self.atomspace.get_high_attention_atoms()
        
        return {
            'total_atoms': len(self.atomspace.atoms),
            'high_attention_atoms': [
                {
                    'name': atom.name,
                    'type': atom.atom_type.value,
                    'sti': atom.attention_value.sti,
                    'lti': atom.attention_value.lti,
                    'truth_strength': atom.truth_value.strength,
                    'truth_confidence': atom.truth_value.confidence
                }
                for atom in high_attention_atoms
            ],
            'attention_distribution': self.atomspace.attention_bank.get_attention_distribution(),
            'agent_performance': {
                agent: sum(performances) / len(performances) if performances else 0
                for agent, performances in self.agent_performance.items()
            },
            'memory_episodes': len(self.memory.episodic_memory),
            'interaction_patterns': dict(list(self.memory.semantic_patterns.items())[:10])
        }

# Global cognitive adapter instance
cognitive_adapter = CognitiveAgentAdapter()