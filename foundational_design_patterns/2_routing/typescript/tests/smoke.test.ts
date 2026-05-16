import { expect, test } from "bun:test";
import { routeQuery } from "../src/routing_basic";
import { cascadeRoute } from "../src/routing_advanced";

test("routing basic classifies billing", () => {
  expect(routeQuery("need refund for payment")).toBe("billing");
});

test("routing advanced classifies engineering", () => {
  expect(cascadeRoute("API latency incident")).toBe("engineering");
});
