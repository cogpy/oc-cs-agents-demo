# OpenCog-Inspired Cognitive Adaptation Integration

This document describes the OpenCog-inspired cognitive adaptation system integrated into the customer service agents demo.

## Overview

The integration implements core concepts from OpenCog's cognitive architecture to enable intelligent adaptation and learning in the customer service system. Instead of static rule-based routing, the system now learns from interactions and dynamically improves its agent selection and response quality.

## Key Concepts Implemented

### 1. AtomSpace-Like Knowledge Representation

The `CognitiveAtomSpace` class implements OpenCog's foundational knowledge representation:

- **Atoms**: Basic units of knowledge with types (ConceptNode, PredicateNode, etc.)
- **Truth Values**: Strength and confidence values for beliefs (0.0 to 1.0 each)
- **Attention Values**: Short-term (STI), long-term (LTI), and very long-term (VLTI) importance
- **Links**: Relationships between atoms (EvaluationLink, InheritanceLink, etc.)

### 2. Attention Allocation Mechanism

Inspired by OpenCog's Attention Allocation, the system manages cognitive resources:

- **Attentional Bank**: Manages total cognitive attention (1000 STI units)
- **Spreading Activation**: Attention spreads to related atoms
- **Decay Mechanisms**: Attention naturally decays over time
- **Dynamic Allocation**: Agents receive attention based on performance

### 3. Memory Systems

Two complementary memory systems store and retrieve interaction knowledge:

#### Episodic Memory
- Stores complete interaction episodes with context, actions, and outcomes
- Enables retrieval of similar past situations
- Supports case-based reasoning for agent selection

#### Semantic Memory  
- Extracts patterns from repeated interactions
- Builds semantic knowledge about action-outcome relationships
- Enables generalization across different contexts

### 4. Learning and Adaptation

The system continuously learns from interactions:

- **Satisfaction Scoring**: Automatically calculates interaction quality
- **Agent Performance Tracking**: Monitors long-term agent effectiveness
- **Context Pattern Learning**: Maps contexts to successful agent choices
- **Cognitive Suggestions**: Provides AI-driven agent selection recommendations

## Technical Implementation

### Core Classes

#### `CognitiveAtomSpace`
```python
class CognitiveAtomSpace:
    """AtomSpace-like knowledge representation system."""
    
    def add_atom(self, name: str, atom_type: AtomType, 
                 truth_value: TruthValue = None,
                 attention_value: AttentionValue = None) -> Atom
    
    def update_attention(self, atom_name: str, sti_boost: float = 10.0)
    
    def get_related_atoms(self, atom_name: str, max_depth: int = 2) -> Set[str]
```

#### `CognitiveMemory`
```python
class CognitiveMemory:
    """Episodic and semantic memory system."""
    
    def store_episode(self, context: Dict, action: str, outcome: str, satisfaction: float)
    
    def retrieve_similar_episodes(self, context: Dict, limit: int = 5) -> List[Dict]
```

#### `CognitiveAgentAdapter`
```python
class CognitiveAgentAdapter:
    """Main cognitive adaptation layer for OpenAI agents."""
    
    def learn_from_interaction(self, agent_name: str, context: Dict, 
                             user_message: str, agent_response: str, 
                             satisfaction: float = 0.5)
    
    def suggest_best_agent(self, context: Dict, user_message: str) -> Dict[str, float]
```

### Integration Points

#### 1. Enhanced Context
The `AirlineAgentContext` now includes cognitive fields:
```python
class AirlineAgentContext(BaseModel):
    # ... existing fields ...
    cognitive_state: dict | None = None
    interaction_count: int = 0
    satisfaction_history: list[float] = []
    learned_preferences: dict[str, float] = {}
```

#### 2. API Enhancements
- **Chat API**: Returns cognitive state and suggestions with each response
- **Cognitive Insights API**: Provides detailed analytics endpoint `/cognitive-insights`
- **Real-time Learning**: Updates cognitive models with each interaction

#### 3. UI Visualization
The new `CognitivePanel` component displays:
- High attention atoms with visual indicators
- Agent performance trends and satisfaction scores
- Memory statistics (episodic episodes, semantic patterns)
- Real-time cognitive suggestions
- Attention distribution across agents

## Cognitive Adaptation Features

### 1. Intelligent Agent Routing
The system suggests optimal agents based on:
- **Historical Performance**: Track which agents succeed in similar contexts
- **Attention Allocation**: Prioritize agents with high cognitive attention
- **Episodic Similarity**: Find agents that worked in similar past situations
- **Semantic Patterns**: Use learned patterns about action-outcome relationships

### 2. Continuous Learning
Every interaction improves the system:
- **Automatic Satisfaction Scoring**: Analyzes response quality automatically
- **Context Feature Extraction**: Identifies key contextual elements
- **Pattern Recognition**: Discovers recurring successful interaction patterns
- **Adaptive Thresholds**: Adjusts decision boundaries based on experience

### 3. Attention Management
OpenCog-inspired attention mechanisms:
- **Resource Allocation**: Limited cognitive attention forces prioritization
- **Spreading Activation**: Related concepts receive attention boosts
- **Temporal Decay**: Older, unused knowledge gradually loses attention
- **Dynamic Reallocation**: Attention shifts based on current needs

### 4. Memory Integration
Dual memory systems work together:
- **Episode Retrieval**: "Remember when a customer with similar context..."
- **Pattern Application**: "Usually customers asking X need agent Y"
- **Similarity Matching**: Find contextually similar past interactions
- **Confidence Assessment**: Use truth values to weight suggestions

## Usage Examples

### 1. Basic Cognitive Learning
```python
# System learns from each interaction
cognitive_adapter.learn_from_interaction(
    agent_name="seat_booking_agent",
    context={"confirmation_number": "ABC123", "has_confirmation": True},
    user_message="I want to change my seat",
    agent_response="I've updated your seat to 23A",
    satisfaction=0.85  # High satisfaction score
)
```

### 2. Agent Selection
```python
# Get cognitive suggestions for agent selection
suggestions = cognitive_adapter.suggest_best_agent(
    context={"confirmation_number": "DEF456"},
    user_message="Can I cancel my flight?"
)
# Returns: {"cancellation_agent": 0.73, "triage_agent": 0.45, ...}
```

### 3. Attention Monitoring
```python
# Check which concepts have high attention
high_attention = atomspace.get_high_attention_atoms(limit=5)
for atom in high_attention:
    print(f"{atom.name}: STI={atom.attention_value.sti}")
```

## Configuration and Tuning

### Attention Parameters
- **Total STI Budget**: 1000 units (configurable)
- **Attention Decay Rate**: 0.95 per cycle (configurable)  
- **Spreading Activation Factor**: 0.3 (configurable)
- **Attention Threshold**: Minimum attention for consideration

### Learning Parameters
- **Satisfaction Calculation**: Weighted factors for response quality
- **Memory Limits**: Episodic (1000 episodes), History (100 interactions)
- **Similarity Threshold**: 0.3 for episode retrieval
- **Pattern Confidence**: Truth value confidence building

### Performance Tuning
- **Update Frequency**: Cognitive state refresh intervals
- **Batch Processing**: Group updates for efficiency
- **Memory Management**: Automatic cleanup of old data
- **Attention Normalization**: Prevent attention inflation

## Benefits of OpenCog Integration

### 1. Adaptive Intelligence
- **Self-Improving**: System gets better with more interactions
- **Context-Aware**: Considers full situational context
- **Personalized**: Adapts to individual user patterns
- **Predictive**: Anticipates user needs based on context

### 2. Cognitive Transparency
- **Explainable Decisions**: Trace why certain agents were suggested
- **Performance Metrics**: Clear visibility into agent effectiveness
- **Learning Progress**: Monitor cognitive development over time
- **Attention Tracking**: Understand what the system finds important

### 3. Robust Performance
- **Graceful Degradation**: Continues working even with limited data
- **Dynamic Adaptation**: Adjusts to changing user patterns
- **Confidence Assessment**: Provides uncertainty estimates
- **Fallback Mechanisms**: Default behavior when confidence is low

## Future Enhancements

### 1. Advanced Reasoning
- **Pattern Logic Networks**: More sophisticated pattern matching
- **Probabilistic Reasoning**: Integrate uncertainty more deeply
- **Causal Learning**: Understand cause-effect relationships
- **Meta-Learning**: Learn how to learn more effectively

### 2. Enhanced Memory
- **Hierarchical Memory**: Multiple time-scale memory systems
- **Associative Networks**: Rich relationship modeling
- **Compression Algorithms**: Efficient long-term storage
- **Forgetting Mechanisms**: Intelligent information pruning

### 3. Multi-Agent Coordination
- **Collective Intelligence**: Agents share learned knowledge
- **Competitive Learning**: Agents compete for cognitive resources
- **Collaborative Filtering**: Cross-agent pattern sharing
- **Emergent Behaviors**: System-level intelligence emergence

## Monitoring and Analytics

The system provides comprehensive monitoring through:

### Real-time Metrics
- **Cognitive State**: Current atomspace statistics
- **Attention Distribution**: How cognitive resources are allocated
- **Agent Performance**: Success rates and trends
- **Memory Utilization**: Episode storage and pattern counts

### Historical Analytics
- **Learning Curves**: Performance improvement over time
- **Pattern Evolution**: How learned patterns develop
- **Attention Trends**: Long-term attention allocation changes
- **Satisfaction Trends**: Overall system effectiveness

### Debug Information
- **Atom Traces**: Follow specific concept evolution
- **Decision Logs**: Why certain agents were selected
- **Learning Events**: When and what the system learned
- **Attention Flows**: How attention spreads through the network

This OpenCog-inspired integration transforms the static customer service system into a dynamic, learning, and adaptive cognitive agent ecosystem that continuously improves its performance through experience.