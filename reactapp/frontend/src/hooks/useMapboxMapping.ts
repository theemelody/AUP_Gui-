import { useCallback, useEffect, useRef } from "react";
import { fetchMapboxCeaUseTypeMapping } from "../services/api.js";

export function useMapboxMapping() {
  const mappingRef = useRef<Record<string, string>>({});
  const promiseRef = useRef<Promise<Record<string, string>> | null>(null);

  const ensureMappingLoaded = useCallback(async (): Promise<Record<string, string>> => {
    if (Object.keys(mappingRef.current).length > 0) return mappingRef.current;

    if (!promiseRef.current) {
      promiseRef.current = fetchMapboxCeaUseTypeMapping()
        .then((mapping: unknown) => {
          if (mapping && typeof mapping === "object") {
            mappingRef.current = mapping as Record<string, string>;
            return mappingRef.current;
          }
          return {};
        })
        .catch(() => ({}))
        .finally(() => { promiseRef.current = null; });
    }

    return promiseRef.current;
  }, []);

  // Warm the cache early to avoid a race on first selection.
  useEffect(() => {
    ensureMappingLoaded();
  }, [ensureMappingLoaded]);

  return { ensureMappingLoaded };
}
