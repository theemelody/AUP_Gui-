import { useCallback, useReducer } from 'react';
import {
  INITIAL_NAVIGATION_STATE,
  navigationReducer,
  type NavigationAction,
  type PageId,
} from '../states/navigationMachine';

export interface UseNavigationResult {
  activePage: PageId;
  navigate: (page: PageId) => void;
}

export function useNavigation(): UseNavigationResult {
  const [state, dispatch] = useReducer(
    navigationReducer,
    INITIAL_NAVIGATION_STATE,
  );

  const navigate = useCallback((page: PageId) => {
    dispatch({ type: 'NAVIGATE', page } satisfies NavigationAction);
  }, []);

  return { activePage: state.activePage, navigate };
}
