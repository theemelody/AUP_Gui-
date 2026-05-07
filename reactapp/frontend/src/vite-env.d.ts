/// <reference types="vite/client" />

declare module 'elkjs/lib/elk.bundled.js' {
  import ELK from 'elkjs';
  export default ELK;
}

declare module 'react-collapse' {
  interface CollapseProps {
    isOpened: boolean;
    children?: import('react').ReactNode;
  }
  export const Collapse: (props: CollapseProps) => import('react').ReactElement | null;
}

declare module '@mapbox/mapbox-gl-draw' {
  import type { IControl } from 'mapbox-gl';
  interface MapboxDrawOptions {
    displayControlsDefault?: boolean;
    controls?: { polygon?: boolean; trash?: boolean; [key: string]: boolean | undefined };
    [key: string]: unknown;
  }
  class MapboxDraw implements IControl {
    constructor(options?: MapboxDrawOptions);
    getAll(): { type: 'FeatureCollection'; features: unknown[] };
    deleteAll(): this;
    onAdd(map: unknown): HTMLElement;
    onRemove(map: unknown): void;
  }
  export default MapboxDraw;
}
