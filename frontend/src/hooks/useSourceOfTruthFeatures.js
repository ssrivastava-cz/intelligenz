import { useEffect, useState } from "react";

import { sourceOfTruthApi } from "../api/sourceOfTruthApi.js";

/**
 * Real feature list for the Test Plan Generator's Feature dropdown,
 * replacing the old hardcoded mock. Loaded once on mount.
 */
export function useSourceOfTruthFeatures() {
  const [features, setFeatures] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    sourceOfTruthApi
      .listFeatures()
      .then((summaries) => {
        if (cancelled) return;
        setFeatures(summaries.map((summary) => ({ value: summary.feature, label: summary.feature })));
        setIsLoading(false);
      })
      .catch((caughtError) => {
        if (cancelled) return;
        setError(caughtError.message);
        setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { features, isLoading, error };
}
