/**
 * 全局类型声明
 */

// 声明 model-viewer 自定义元素
declare namespace JSX {
  interface IntrinsicElements {
    'model-viewer': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement> & {
      src?: string;
      alt?: string;
      'auto-rotate'?: boolean;
      'camera-controls'?: boolean;
      style?: React.CSSProperties;
    }, HTMLElement>;
  }
}

