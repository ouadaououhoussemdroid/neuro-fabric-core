/**
 * Tiny structured logger. Emits one JSON line per event to console so it shows
 * up in Worker / preview logs. No external sink.
 */
export type LogLevel = "debug" | "info" | "warn" | "error";

export interface LogEvent {
  level: LogLevel;
  msg: string;
  ts: string;
  [k: string]: unknown;
}

export function log(level: LogLevel, msg: string, fields: Record<string, unknown> = {}): void {
  const evt: LogEvent = { level, msg, ts: new Date().toISOString(), ...fields };
  const line = JSON.stringify(evt);
  if (level === "error") console.error(line);
  else if (level === "warn") console.warn(line);
  else console.log(line);
}

export interface Timer {
  end(extra?: Record<string, unknown>): number;
}

/** Start a named timer that emits a single log line on end(). */
export function startTimer(name: string, fields: Record<string, unknown> = {}): Timer {
  const t0 = performance.now();
  return {
    end(extra: Record<string, unknown> = {}) {
      const ms = +(performance.now() - t0).toFixed(2);
      log("info", `timing.${name}`, { ...fields, ...extra, durationMs: ms });
      return ms;
    },
  };
}
