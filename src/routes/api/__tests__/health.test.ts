import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the Supabase admin client.
const mockRpc = vi.fn();
const mockSupabaseAdmin = {
  rpc: mockRpc,
};

vi.mock("@/integrations/supabase/client.server", () => ({
  supabaseAdmin: mockSupabaseAdmin,
}));

// Mock the ONNX adapter.
const mockIsONNX = vi.fn();
const mockResetProbe = vi.fn();
vi.mock("@/lib/ai/adapters/onnx-adapter", () => ({
  isONNXRuntimeAvailable: (...args: unknown[]) => mockIsONNX(...args),
  __resetONNXCapabilityProbe: (...args: unknown[]) => mockResetProbe(...args),
}));

// Mock logging.
vi.mock("@/lib/logging", () => ({
  log: vi.fn(),
}));

// Import after mocks.
const { Route } = await import("../health");

type GetHandler = (ctx: { request: Request; context: unknown }) => Promise<Response>;

function callHealth() {
  const handlers = Route.options.server!.handlers as unknown as { GET: GetHandler };
  return handlers.GET({ request: new Request("http://localhost/api/health"), context: {} });
}

describe("GET /api/health", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Default: everything healthy.
    mockRpc.mockResolvedValue({ data: [{ ok: true }], error: null });
    mockIsONNX.mockResolvedValue(true);
    mockResetProbe.mockClear();
  });

  it("returns 200 with ok status when all checks pass", async () => {
    const res = await callHealth();
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("ok");
    expect(body.version).toBeDefined();
    expect(body.timestamp).toBeDefined();
    expect(body.checks).toEqual([
      { name: "application", status: "ok" },
      { name: "database", status: "ok" },
      { name: "onnx_runtime", status: "ok" },
    ]);
  });

  it("returns 503 when database is unreachable (RPC returns error)", async () => {
    mockRpc.mockResolvedValue({ data: null, error: { message: "connection refused" } });
    const res = await callHealth();
    expect(res.status).toBe(503);
    const body = await res.json();
    expect(body.status).toBe("down");
    expect(
      body.checks.some(
        (c: { name: string; status: string }) => c.name === "database" && c.status === "down",
      ),
    ).toBe(true);
  });

  it("returns 503 when database RPC returns no data", async () => {
    mockRpc.mockResolvedValue({ data: [{ ok: false }], error: null });
    const res = await callHealth();
    expect(res.status).toBe(503);
    const body = await res.json();
    expect(body.status).toBe("down");
    expect(
      body.checks.some(
        (c: { name: string; status: string }) => c.name === "database" && c.status === "down",
      ),
    ).toBe(true);
  });

  it("returns 200 with degraded status when ONNX runtime is unavailable", async () => {
    mockIsONNX.mockResolvedValue(false);
    const res = await callHealth();
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("degraded");
    expect(
      body.checks.some(
        (c: { name: string; status: string }) =>
          c.name === "onnx_runtime" && c.status === "degraded",
      ),
    ).toBe(true);
    expect(
      body.checks.some(
        (c: { name: string; status: string }) => c.name === "database" && c.status === "ok",
      ),
    ).toBe(true);
  });

  it("does not require authentication", async () => {
    const res = await callHealth();
    expect(res.status).toBe(200);
  });

  it("resets the ONNX capability probe before checking", async () => {
    await callHealth();
    expect(mockResetProbe).toHaveBeenCalled();
  });

  it("returns 503 when supabaseAdmin RPC throws an exception", async () => {
    mockRpc.mockRejectedValue(new Error("network timeout"));
    const res = await callHealth();
    expect(res.status).toBe(503);
    const body = await res.json();
    expect(body.status).toBe("down");
  });
});
