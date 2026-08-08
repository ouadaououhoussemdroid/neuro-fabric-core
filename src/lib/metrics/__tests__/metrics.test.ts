import { describe, it, expect, beforeEach } from "vitest";
import { Counter, Gauge, Histogram, renderPrometheusMetrics, resetMetrics } from "../index";

describe("Metrics primitives", () => {
  beforeEach(() => {
    resetMetrics();
  });

  describe("Counter", () => {
    it("increments by 1 by default", () => {
      const c = new Counter("test_counter", "A test counter");
      expect(c.value()).toBe(0);
      c.inc();
      expect(c.value()).toBe(1);
      c.inc();
      expect(c.value()).toBe(2);
    });

    it("increments by a custom amount", () => {
      const c = new Counter("test_counter2", "Another test counter");
      c.inc({}, 5);
      expect(c.value()).toBe(5);
    });

    it("tracks per-label values", () => {
      const c = new Counter("test_counter3", "Labeled counter");
      c.inc({ method: "upload" });
      c.inc({ method: "upload" });
      c.inc({ method: "download" });
      expect(c.value({ method: "upload" })).toBe(2);
      expect(c.value({ method: "download" })).toBe(1);
      expect(c.value()).toBe(3);
    });
  });

  describe("Gauge", () => {
    it("sets absolute values", () => {
      const g = new Gauge("test_gauge", "A test gauge");
      g.set({}, 42);
      expect(g.value()).toBe(42);
      g.set({}, 17);
      expect(g.value()).toBe(17);
    });

    it("can increment and decrement", () => {
      const g = new Gauge("test_gauge2", "Another gauge");
      g.inc();
      g.inc();
      expect(g.value()).toBe(2);
      g.dec({}, 1);
      expect(g.value()).toBe(1);
    });

    it("tracks per-label values", () => {
      const g = new Gauge("test_gauge3", "Labeled gauge");
      g.set({ component: "a" }, 10);
      g.set({ component: "b" }, 20);
      expect(g.value({ component: "a" })).toBe(10);
      expect(g.value({ component: "b" })).toBe(20);
    });
  });

  describe("Histogram", () => {
    it("records observations and counts them", () => {
      const h = new Histogram("test_hist", "A test histogram");
      h.observe({}, 10);
      h.observe({}, 50);
      h.observe({}, 100);
    });

    it("tracks per-label histograms", () => {
      const h = new Histogram("test_hist2", "Labeled histogram");
      h.observe({ model: "eegconformer" }, 30);
      h.observe({ model: "pca" }, 5);
    });
  });

  describe("renderPrometheusMetrics", () => {
    it("renders counter in Prometheus text format", () => {
      const c = new Counter("my_counter", "My counter");
      c.inc({}, 3);
      const output = renderPrometheusMetrics();
      expect(output).toContain("# HELP my_counter My counter");
      expect(output).toContain("# TYPE my_counter counter");
      expect(output).toContain("my_counter 3");
    });

    it("renders gauge in Prometheus text format", () => {
      const g = new Gauge("my_gauge", "My gauge");
      g.set({}, 7);
      const output = renderPrometheusMetrics();
      expect(output).toContain("# HELP my_gauge My gauge");
      expect(output).toContain("# TYPE my_gauge gauge");
      expect(output).toContain("my_gauge 7");
    });

    it("renders histogram with buckets, sum, and count", () => {
      const h = new Histogram("my_histogram", "My histogram");
      h.observe({}, 15);
      h.observe({}, 25);
      const output = renderPrometheusMetrics();
      expect(output).toContain("# HELP my_histogram My histogram");
      expect(output).toContain("# TYPE my_histogram histogram");
      expect(output).toContain('_bucket{le="5"');
      expect(output).toContain('_bucket{le="25"');
      expect(output).toContain('_bucket{le="10000"}');
      expect(output).toContain("_sum 40");
      expect(output).toContain("_count 2");
    });

    it("renders labeled metrics with label sets", () => {
      const c = new Counter("labeled_counter", "A labeled counter");
      c.inc({ status: "200" }, 5);
      c.inc({ status: "500" }, 1);
      const output = renderPrometheusMetrics();
      expect(output).toContain('labeled_counter{status="200"} 5');
      expect(output).toContain('labeled_counter{status="500"} 1');
    });
  });
});
