import type { ChatResponse, CognitiveInsights } from "./types";

const API_BASE_URL = "http://localhost:8000";

export async function callChatAPI(
  message: string,
  conversationId: string,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      conversation_id: conversationId || null,
    }),
  });

  if (!response.ok) {
    throw new Error(`API call failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getCognitiveInsights(): Promise<CognitiveInsights> {
  const response = await fetch(`${API_BASE_URL}/cognitive-insights`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`API call failed: ${response.statusText}`);
  }

  return response.json();
}
