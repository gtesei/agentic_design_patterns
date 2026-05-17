import { describe, expect, test } from "bun:test";
import { fetchWikipediaSummary } from "../src/computer_use_basic.ts";
import { runPlaywrightTask } from "../src/computer_use_advanced.ts";

describe("computer_use (TS) — module shape", () => {
  test("playwright task is exported", async () => {
    expect(await runPlaywrightTask()).toBeString();
  });

  test("fetchWikipediaSummary is a function", () => {
    expect(typeof fetchWikipediaSummary).toBe("function");
  });
});
