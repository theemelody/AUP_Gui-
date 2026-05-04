export type PageId = 'simulation' | 'tech-tree' | 'kpi' | 'secap';

export interface NavigationState {
  activePage: PageId;
}

export type NavigationAction = { type: 'NAVIGATE'; page: PageId };

export const INITIAL_NAVIGATION_STATE: NavigationState = {
  activePage: 'simulation',
};

export function navigationReducer(
  state: NavigationState,
  action: NavigationAction,
): NavigationState {
  switch (action.type) {
    case 'NAVIGATE':
      if (state.activePage === action.page) return state;
      return { activePage: action.page };
    default:
      return state;
  }
}
