/**
 * Ambient type stub for the optional `brainflow` Node binding.
 *
 * The real package is not in package.json (it requires a platform-specific
 * native install). This stub lets `import("brainflow")` typecheck without
 * the dependency; the adapter (T-005) handles the case where the runtime
 * import fails.
 */
declare module "brainflow" {
  export class BoardShim {
    constructor(boardId: string | number, serialPort?: string);
    prepare_session(): void;
    start_stream(numSamples?: number): void;
    stop_stream(): void;
    get_board_data(numSamples: number): number[][];
    release_session(): void;
  }
  const _default: { BoardShim: typeof BoardShim };
  export default _default;
}
