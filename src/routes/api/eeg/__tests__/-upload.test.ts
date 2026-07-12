import { describe, it, expect, vi, beforeEach } from "vitest";

const mockAuthenticateRequest = vi.fn();
const mockCheckRateLimit = vi.fn();

vi.mock("@/integrations/supabase/request-auth", async () => {
  const actual = await vi.importActual<typeof import("@/integrations/supabase/request-auth")>(
    "@/integrations/supabase/request-auth",
  );
  return {
    ...actual,
    authenticateRequest: (...args: unknown[]) => mockAuthenticateRequest(...args),
  };
});

vi.mock("@/integrations/supabase/rate-limit", () => ({
  checkRateLimit: (...args: unknown[]) => mockCheckRateLimit(...args),
}));

// Import after the mocks are registered.
const { Route } = await import("../upload");
const { AuthError } = await import("@/integrations/supabase/request-auth");

function fakeSupabase(insertResult: { data: unknown; error: unknown }) {
  const single = vi.fn().mockResolvedValue(insertResult);
  const select = vi.fn(() => ({ single }));
  const insert = vi.fn(() => ({ select }));
  const from = vi.fn(() => ({ insert }));
  return { from } as unknown as Parameters<typeof mockAuthenticateRequest>[0];
}

type PostHandler = (ctx: { request: Request; context: unknown }) => Promise<Response>;

function callUpload(request: Request) {
  const handlers = Route.options.server!.handlers as unknown as { POST: PostHandler };
  return handlers.POST({ request, context: {} });
}

function csvFile(name = "signal.csv", channels = 3, rows = 500): File {
  const lines: string[] = [];
  for (let r = 0; r < rows; r++) {
    const cells: string[] = [];
    for (let c = 0; c < channels; c++) cells.push(String(Math.sin((r + c) * 0.1)));
    lines.push(cells.join(","));
  }
  return new File([lines.join("\n")], name, { type: "text/csv" });
}

function uploadRequest(opts: { form: FormData; auth?: boolean }): Request {
  const headers: Record<string, string> = {};
  if (opts.auth !== false) headers.authorization = "Bearer test-token";
  // FormData bodies set their own multipart content-type with boundary;
  // Request does this automatically when body is a FormData instance.
  return new Request("http://localhost/api/eeg/upload", {
    method: "POST",
    body: opts.form,
    headers,
  });
}

describe("POST /api/eeg/upload", () => {
  beforeEach(() => {
    mockAuthenticateRequest.mockReset();
    mockCheckRateLimit.mockReset();
    mockCheckRateLimit.mockResolvedValue({ allowed: true, retryAfterMs: 0 });
  });

  it("returns 401 when authenticateRequest rejects", async () => {
    mockAuthenticateRequest.mockRejectedValue(
      new AuthError("Unauthorized: missing Bearer token", 401),
    );
    const form = new FormData();
    form.set("file", csvFile());
    form.set("sampleRate", "128");
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toMatch(/Unauthorized/);
  });

  it("propagates a non-401 AuthError status (e.g. 500 for server misconfiguration)", async () => {
    mockAuthenticateRequest.mockRejectedValue(new AuthError("Server misconfigured", 500));
    const form = new FormData();
    form.set("file", csvFile());
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(500);
  });

  it("returns 429 when the rate limit is exceeded", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "a1" }, error: null }),
    });
    mockCheckRateLimit.mockResolvedValue({ allowed: false, retryAfterMs: 12345 });
    const form = new FormData();
    form.set("file", csvFile());
    form.set("sampleRate", "128");
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.retry_after_ms).toBe(12345);
  });

  it("fails open (200s through) when the rate-limit check itself throws", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "a1" }, error: null }),
    });
    mockCheckRateLimit.mockRejectedValue(new Error("db unreachable"));
    const form = new FormData();
    form.set("file", csvFile());
    form.set("sampleRate", "128");
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(200);
  });

  it("returns 400 when the file field is missing", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "a1" }, error: null }),
    });
    const form = new FormData();
    form.set("notAFile", "oops");
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(400);
  });

  it("returns 413 when the file exceeds the 50MB size cap", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "a1" }, error: null }),
    });
    const big = new File([new Uint8Array(50 * 1024 * 1024 + 1)], "big.csv");
    const form = new FormData();
    form.set("file", big);
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(413);
  });

  it("returns 400 for an empty file", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "a1" }, error: null }),
    });
    const form = new FormData();
    form.set("file", new File([], "empty.csv"));
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(400);
  });

  it("returns 415 for an unsupported file extension", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "a1" }, error: null }),
    });
    const form = new FormData();
    form.set("file", new File(["hello"], "signal.xyz"));
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(415);
  });

  it("returns 400 when sampleRate is missing for a CSV file", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "a1" }, error: null }),
    });
    const form = new FormData();
    form.set("file", csvFile());
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toMatch(/sampleRate required/);
  });

  it("returns 422 when the file fails to parse", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "a1" }, error: null }),
    });
    const form = new FormData();
    form.set("file", new File(["not a valid edf"], "signal.edf"));
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(422);
  });

  it("processes a valid CSV end-to-end and returns 200 with a full analysis payload", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "analysis-123" }, error: null }),
    });
    const form = new FormData();
    form.set("file", csvFile("signal.csv", 3, 600));
    form.set("sampleRate", "128");
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.analysis_id).toBe("analysis-123");
    expect(body.persisted).toBe(true);
    expect(Array.isArray(body.embedding)).toBe(true);
    expect(body.dimensions).toBeGreaterThan(0);
    expect(body.decoder.attention).toBeGreaterThanOrEqual(0);
    expect(body.signal.channels).toHaveLength(3);
    expect(body.timings.total_ms).toBeGreaterThanOrEqual(0);
  });

  it("still returns 200 with persisted=false when the DB insert fails (does not fail the request)", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: null, error: { message: "insert failed" } }),
    });
    const form = new FormData();
    form.set("file", csvFile("signal.csv", 3, 600));
    form.set("sampleRate", "128");
    const res = await callUpload(uploadRequest({ form }));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.persisted).toBe(false);
    expect(body.analysis_id).toBeNull();
  });

  it("returns 400 for a non-multipart content-type", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase({ data: { id: "a1" }, error: null }),
    });
    const res = await callUpload(
      new Request("http://localhost/api/eeg/upload", {
        method: "POST",
        headers: { authorization: "Bearer x", "content-type": "application/json" },
        body: "{}",
      }),
    );
    expect(res.status).toBe(400);
  });
});
