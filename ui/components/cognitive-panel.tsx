"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, Zap, TrendingUp, Database, Activity } from "lucide-react";
import { getCognitiveInsights } from "@/lib/api";
import type { CognitiveInsights, CognitiveState } from "@/lib/types";

interface CognitivePanelProps {
  cognitiveState?: CognitiveState;
  cognitiveSuggestions?: Record<string, number>;
}

export function CognitivePanel({ 
  cognitiveState, 
  cognitiveSuggestions 
}: CognitivePanelProps) {
  const [insights, setInsights] = useState<CognitiveInsights | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchInsights = async () => {
    setIsLoading(true);
    try {
      const data = await getCognitiveInsights();
      setInsights(data);
    } catch (error) {
      console.error("Failed to fetch cognitive insights:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchInsights();
    const interval = setInterval(fetchInsights, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getAttentionColor = (sti: number) => {
    if (sti > 50) return "bg-red-500";
    if (sti > 20) return "bg-yellow-500";
    return "bg-green-500";
  };

  const getSatisfactionColor = (score: number) => {
    if (score > 0.7) return "text-green-600";
    if (score > 0.5) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="w-80 bg-white border-l border-gray-200 p-4 overflow-y-auto">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="h-6 w-6 text-purple-600" />
        <h2 className="text-lg font-semibold">OpenCog Cognitive Insights</h2>
      </div>

      {/* Current Cognitive State */}
      {cognitiveState && (
        <Card className="mb-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Current State
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Total Atoms:</span>
              <Badge variant="outline">{cognitiveState.total_atoms}</Badge>
            </div>
            <div className="flex justify-between text-sm">
              <span>Memory Episodes:</span>
              <Badge variant="outline">{cognitiveState.memory_episodes}</Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Cognitive Suggestions */}
      {cognitiveSuggestions && Object.keys(cognitiveSuggestions).length > 0 && (
        <Card className="mb-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Agent Suggestions
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {Object.entries(cognitiveSuggestions)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 3)
              .map(([agent, score]) => (
                <div key={agent} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="capitalize">{agent.replace('_', ' ')}</span>
                    <span className="font-mono text-xs">
                      {score.toFixed(3)}
                    </span>
                  </div>
                  <Progress value={score * 100} className="h-2" />
                </div>
              ))}
          </CardContent>
        </Card>
      )}

      {/* High Attention Atoms */}
      {cognitiveState?.high_attention_atoms && (
        <Card className="mb-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Database className="h-4 w-4" />
              High Attention Atoms
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {cognitiveState.high_attention_atoms.slice(0, 5).map((atom, index) => (
              <div key={index} className="p-2 bg-gray-50 rounded text-xs">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium truncate">{atom.name}</span>
                  <Badge variant="secondary" className="text-xs">
                    {atom.type}
                  </Badge>
                </div>
                <div className="flex gap-2">
                  <div className="flex items-center gap-1">
                    <div className={`w-2 h-2 rounded ${getAttentionColor(atom.sti)}`} />
                    <span>STI: {atom.sti.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span>TV: {atom.truth_strength.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Agent Performance from Insights */}
      {insights?.agent_performance && (
        <Card className="mb-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Agent Performance
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {Object.entries(insights.agent_performance).map(([agent, perf]) => (
              <div key={agent} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium capitalize">
                    {agent.replace('_agent', '').replace('_', ' ')}
                  </span>
                  <Badge 
                    variant="outline" 
                    className={getSatisfactionColor(perf.average_satisfaction)}
                  >
                    {(perf.average_satisfaction * 100).toFixed(0)}%
                  </Badge>
                </div>
                <div className="text-xs text-gray-600">
                  <div className="flex justify-between">
                    <span>Interactions: {perf.total_interactions}</span>
                    <span>Trend: {(perf.recent_trend * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <Progress 
                  value={perf.average_satisfaction * 100} 
                  className="h-1" 
                />
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Memory Statistics */}
      {insights?.memory_statistics && (
        <Card className="mb-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Database className="h-4 w-4" />
              Memory Statistics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Episodic Memory:</span>
              <Badge variant="outline">
                {insights.memory_statistics.episodic_memory_size}
              </Badge>
            </div>
            <div className="flex justify-between text-sm">
              <span>Semantic Patterns:</span>
              <Badge variant="outline">
                {insights.memory_statistics.semantic_patterns_count}
              </Badge>
            </div>
            <div className="flex justify-between text-sm">
              <span>Recent Interactions:</span>
              <Badge variant="outline">
                {insights.memory_statistics.recent_interactions}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="text-xs text-gray-500 mt-4">
        <div className="flex items-center gap-1 mb-2">
          <div className="w-2 h-2 bg-red-500 rounded" />
          <span>High Attention (STI &gt; 50)</span>
        </div>
        <div className="flex items-center gap-1 mb-2">
          <div className="w-2 h-2 bg-yellow-500 rounded" />
          <span>Medium Attention (STI 20-50)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded" />
          <span>Low Attention (STI &lt; 20)</span>
        </div>
        {isLoading && (
          <div className="mt-2 text-center">
            <span className="animate-pulse">Updating insights...</span>
          </div>
        )}
      </div>
    </div>
  );
}