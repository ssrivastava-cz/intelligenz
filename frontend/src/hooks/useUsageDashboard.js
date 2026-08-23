import { useCallback, useEffect, useState } from "react";

import { indexingHistoryApi } from "../api/indexingHistoryApi.js";
import { usageApi } from "../api/usageApi.js";

/**
 * Loads everything the Usage Dashboard needs — unified AI Generation
 * usage (summary + per-generation records combining both the Test Plan
 * Generator and the Knowledge Assistant, via `usageApi`) and Document
 * Indexing & Embedding usage (stats + list, via `indexingHistoryApi`),
 * two separate AI activity types fetched in parallel. Read-only: never
 * regenerates AI output, never calls OpenAI, never re-embeds anything —
 * both sources are already-persisted history.
 */
export function useUsageDashboard() {
  const [generations, setGenerations] = useState([]);
  const [stats, setStats] = useState(null);
  const [indexingRuns, setIndexingRuns] = useState([]);
  const [indexingStats, setIndexingStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [usageResponse, indexingListResponse, indexingStatsResponse] = await Promise.all([
        usageApi.get(),
        indexingHistoryApi.list(),
        indexingHistoryApi.stats(),
      ]);
      setGenerations(usageResponse.generations);
      setStats(usageResponse.summary);
      setIndexingRuns(indexingListResponse);
      setIndexingStats(indexingStatsResponse);
    } catch (caughtError) {
      setError(caughtError.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { generations, stats, indexingRuns, indexingStats, isLoading, error, refresh: load };
}
