"""
Mock agent system for testing OpenCog cognitive adaptation without OpenAI API.
This allows us to demonstrate the cognitive features without needing valid API keys.
"""

from __future__ import annotations
import random
import asyncio
from typing import Dict, List, Any
from main import (
    AirlineAgentContext, 
    update_cognitive_context, 
    get_cognitive_agent_suggestion,
    calculate_interaction_satisfaction
)

class MockAgent:
    """Mock agent that simulates responses without calling OpenAI API."""
    
    def __init__(self, name: str, specialties: List[str]):
        self.name = name
        self.specialties = specialties
        
    async def respond(self, message: str, context: AirlineAgentContext) -> str:
        """Generate a mock response based on message content and agent specialty."""
        message_lower = message.lower()
        
        # Simulate different response patterns based on agent type
        if "seat" in self.name.lower():
            if any(word in message_lower for word in ["seat", "change", "move"]):
                new_seat = f"{random.randint(1, 30)}{random.choice(['A', 'B', 'C', 'D', 'E', 'F'])}"
                context.seat_number = new_seat
                return f"I've successfully updated your seat to {new_seat}. Is there anything else I can help you with?"
            else:
                return "I can help you with seat changes. Would you like to select a new seat?"
                
        elif "cancellation" in self.name.lower():
            if any(word in message_lower for word in ["cancel", "refund"]):
                return f"I've successfully cancelled your flight {context.flight_number} with confirmation {context.confirmation_number}. You'll receive a refund within 7-10 business days."
            else:
                return "I can help you cancel your flight. Are you sure you want to proceed with cancellation?"
                
        elif "status" in self.name.lower():
            if any(word in message_lower for word in ["status", "flight", "gate", "time"]):
                return f"Your flight {context.flight_number or 'FLT-123'} is on time and departing from gate A10 at the scheduled time."
            else:
                return "I can provide flight status information. What would you like to know?"
                
        elif "faq" in self.name.lower():
            if "baggage" in message_lower:
                return "You're allowed one carry-on bag (22x14x9 inches) and one checked bag up to 50 pounds. Overweight fees apply for bags over 50 pounds."
            elif "wifi" in message_lower:
                return "We offer complimentary WiFi on all flights. Connect to 'Airline-WiFi' network once airborne."
            elif "seat" in message_lower and ("how many" in message_lower or "total" in message_lower):
                return "This aircraft has 120 total seats: 22 business class and 98 economy. Exit rows are 4 and 16, with Economy Plus in rows 5-8."
            else:
                return "I can answer questions about baggage, WiFi, aircraft information, and airline policies. What would you like to know?"
                
        else:  # Triage agent
            # Determine which specialized agent should handle this
            if any(word in message_lower for word in ["seat", "change seat", "move"]):
                return "I'll transfer you to our seat booking specialist who can help you change your seat assignment."
            elif any(word in message_lower for word in ["cancel", "cancellation", "refund"]):
                return "I'll connect you with our cancellation specialist who can help process your flight cancellation."
            elif any(word in message_lower for word in ["status", "flight status", "gate", "delayed"]):
                return "Let me transfer you to our flight status agent who can provide current information about your flight."
            elif any(word in message_lower for word in ["baggage", "wifi", "how many", "policies"]):
                return "I'll connect you with our FAQ specialist who can answer your questions about airline policies and services."
            else:
                return "Hello! I'm here to help with your airline needs. Are you looking to change your seat, check flight status, cancel a booking, or do you have questions about our services?"

class MockAgentSystem:
    """Mock system that simulates the OpenAI Agents SDK behavior."""
    
    def __init__(self):
        self.agents = {
            "Triage Agent": MockAgent("Triage Agent", ["routing", "general"]),
            "Seat Booking Agent": MockAgent("Seat Booking Agent", ["seats", "assignments"]),
            "Flight Status Agent": MockAgent("Flight Status Agent", ["status", "gates", "times"]),
            "Cancellation Agent": MockAgent("Cancellation Agent", ["cancellations", "refunds"]),
            "FAQ Agent": MockAgent("FAQ Agent", ["policies", "information"])
        }
        self.contexts: Dict[str, AirlineAgentContext] = {}
        
    def get_or_create_context(self, conversation_id: str) -> AirlineAgentContext:
        """Get existing context or create new one."""
        if conversation_id not in self.contexts:
            ctx = AirlineAgentContext()
            ctx.account_number = str(random.randint(10000000, 99999999))
            ctx.flight_number = f"FLT-{random.randint(100, 999)}"
            ctx.confirmation_number = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
            ctx.cognitive_state = {}
            self.contexts[conversation_id] = ctx
        return self.contexts[conversation_id]
        
    async def process_message(self, message: str, conversation_id: str) -> Dict[str, Any]:
        """Process a message and return cognitive-enhanced response."""
        context = self.get_or_create_context(conversation_id)
        
        # Get cognitive suggestions for agent selection
        cognitive_suggestions = get_cognitive_agent_suggestion(context, message)
        
        # Determine which agent should respond (simulate smart routing)
        selected_agent = self._select_agent(message, cognitive_suggestions)
        agent = self.agents[selected_agent]
        
        # Generate response
        response = await agent.respond(message, context)
        
        # Calculate satisfaction score
        satisfaction = calculate_interaction_satisfaction(selected_agent, message, response, context)
        
        # Update cognitive learning
        update_cognitive_context(context, selected_agent, message, response, satisfaction)
        
        # Simulate some events for the UI
        events = [
            {
                "id": f"event_{random.randint(1000, 9999)}",
                "type": "message",
                "agent": selected_agent,
                "content": response,
                "timestamp": 1000 * __import__('time').time()
            }
        ]
        
        # Add tool call event if appropriate
        if "updated" in response.lower() or "cancelled" in response.lower():
            events.insert(0, {
                "id": f"tool_{random.randint(1000, 9999)}",
                "type": "tool_call", 
                "agent": selected_agent,
                "content": "update_seat" if "seat" in response.lower() else "cancel_flight",
                "timestamp": 1000 * __import__('time').time()
            })
            
        # Add handoff event if agent changed
        if selected_agent != "Triage Agent" and "transfer" in response.lower():
            events.insert(0, {
                "id": f"handoff_{random.randint(1000, 9999)}",
                "type": "handoff",
                "agent": "Triage Agent", 
                "content": f"Triage Agent -> {selected_agent}",
                "timestamp": 1000 * __import__('time').time()
            })
        
        return {
            "conversation_id": conversation_id,
            "current_agent": selected_agent,
            "messages": [{"content": response, "agent": selected_agent}],
            "events": events,
            "context": context.model_dump(),
            "agents": self._build_agents_list(),
            "guardrails": [],  # No guardrail issues in mock mode
            "cognitive_state": context.cognitive_state,
            "cognitive_suggestions": cognitive_suggestions
        }
        
    def _select_agent(self, message: str, cognitive_suggestions: Dict[str, float]) -> str:
        """Select appropriate agent based on message content and cognitive suggestions."""
        message_lower = message.lower()
        
        # If we have strong cognitive suggestions, use them
        if cognitive_suggestions:
            max_score = max(cognitive_suggestions.values())
            if max_score > 0.5:  # High confidence threshold
                best_agent = max(cognitive_suggestions.items(), key=lambda x: x[1])[0]
                # Convert internal names to display names
                agent_mapping = {
                    "seat_booking_agent": "Seat Booking Agent",
                    "cancellation_agent": "Cancellation Agent", 
                    "flight_status_agent": "Flight Status Agent",
                    "faq_agent": "FAQ Agent",
                    "triage_agent": "Triage Agent"
                }
                if best_agent in agent_mapping:
                    return agent_mapping[best_agent]
        
        # Fallback to rule-based selection
        if any(word in message_lower for word in ["seat", "change seat", "move"]):
            return "Seat Booking Agent"
        elif any(word in message_lower for word in ["cancel", "cancellation", "refund"]):
            return "Cancellation Agent"
        elif any(word in message_lower for word in ["status", "flight status", "gate", "delayed"]):
            return "Flight Status Agent"
        elif any(word in message_lower for word in ["baggage", "wifi", "how many", "policies"]):
            return "FAQ Agent"
        else:
            return "Triage Agent"
            
    def _build_agents_list(self) -> List[Dict[str, Any]]:
        """Build agents list for UI."""
        return [
            {
                "name": "Triage Agent",
                "description": "Routes customer requests to appropriate specialists",
                "handoffs": ["Seat Booking Agent", "Cancellation Agent", "Flight Status Agent", "FAQ Agent"],
                "tools": [],
                "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"]
            },
            {
                "name": "Seat Booking Agent", 
                "description": "Handles seat changes and assignments",
                "handoffs": ["Triage Agent"],
                "tools": ["update_seat", "display_seat_map"],
                "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"]
            },
            {
                "name": "Cancellation Agent",
                "description": "Processes flight cancellations and refunds", 
                "handoffs": ["Triage Agent"],
                "tools": ["cancel_flight"],
                "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"]
            },
            {
                "name": "Flight Status Agent",
                "description": "Provides flight status and gate information",
                "handoffs": ["Triage Agent"],
                "tools": ["flight_status_tool"],
                "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"]
            },
            {
                "name": "FAQ Agent",
                "description": "Answers questions about policies and services",
                "handoffs": ["Triage Agent"], 
                "tools": ["faq_lookup_tool"],
                "input_guardrails": ["Relevance Guardrail", "Jailbreak Guardrail"]
            }
        ]

# Global instance
mock_system = MockAgentSystem()