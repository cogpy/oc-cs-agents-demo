export interface Message {
  id: string
  content: string
  role: "user" | "assistant"
  agent?: string
  timestamp: Date
}

export interface Agent {
  name: string
  description: string
  handoffs: string[]
  tools: string[]
  /** List of input guardrail identifiers for this agent */
  input_guardrails: string[]
}

export type EventType = "message" | "handoff" | "tool_call" | "tool_output" | "context_update"

export interface AgentEvent {
  id: string
  type: EventType
  agent: string
  content: string
  timestamp: Date
  metadata?: {
    source_agent?: string
    target_agent?: string
    tool_name?: string
    tool_args?: Record<string, any>
    tool_result?: any
    context_key?: string
    context_value?: any
    changes?: Record<string, any>
  }
}

export interface GuardrailCheck {
  id: string
  name: string
  input: string
  reasoning: string
  passed: boolean
  timestamp: Date
}

export interface ChatResponse {
  conversation_id: string;
  current_agent: string;
  messages: { content: string; agent: string }[];
  events: AgentEvent[];
  context: Record<string, any>;
  agents: Agent[];
  guardrails?: GuardrailCheck[];
  cognitive_state?: CognitiveState;
  cognitive_suggestions?: Record<string, number>;
}

export interface CognitiveState {
  total_atoms: number;
  high_attention_atoms: AttentionAtom[];
  attention_distribution: Record<string, number>;
  agent_performance: Record<string, number>;
  memory_episodes: number;
  interaction_patterns: Record<string, number>;
}

export interface AttentionAtom {
  name: string;
  type: string;
  sti: number;
  lti: number;
  truth_strength: number;
  truth_confidence: number;
}

export interface CognitiveInsights {
  cognitive_state: CognitiveState;
  attention_distribution: Record<string, number>;
  memory_statistics: {
    episodic_memory_size: number;
    semantic_patterns_count: number;
    recent_interactions: number;
  };
  agent_performance: Record<string, {
    average_satisfaction: number;
    total_interactions: number;
    recent_trend: number;
  }>;
  context_patterns: Record<string, Record<string, number>>;
}

