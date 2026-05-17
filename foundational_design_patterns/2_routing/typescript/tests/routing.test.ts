import { describe, expect, test } from "bun:test";
import {
  bookingHandler,
  coordinatorAgent,
  extractRequest,
  infoHandler,
  unclearHandler,
} from "../src/routing.ts";

describe("routing (TS) — module shape", () => {
  test("exports handlers", () => {
    expect(bookingHandler("x")).toContain("Booking Handler");
    expect(infoHandler("x")).toContain("Info Handler");
    expect(unclearHandler("x")).toContain("Please clarify");
  });

  test("extractRequest returns the plain request string", () => {
    expect(extractRequest({ request: "Book me a flight." })).toBe("Book me a flight.");
  });

  test("coordinatorAgent is invokable", () => {
    expect(typeof coordinatorAgent.invoke).toBe("function");
  });
});
