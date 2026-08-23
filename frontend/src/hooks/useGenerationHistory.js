import { useCallback, useEffect, useState } from "react";

import { generationHistoryApi } from "../api/generationHistoryApi.js";

/**
 * Loads the Generation History list and dashboard stats together. Both
 * requests are read-only; nothing here regenerates AI output or calls
 * OpenAI.
 */
export function useGenerationHistory() {
  const [generations, setGenerations] = useState([]);
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [listResponse, statsResponse] = await Promise.all([
        generationHistoryApi.list(),
        generationHistoryApi.stats(),
      ]);
      setGenerations(listResponse.items);
      setStats(statsResponse);
    } catch (caughtError) {
      setError(caughtError.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { generations, stats, isLoading, error, refresh: load };
}
