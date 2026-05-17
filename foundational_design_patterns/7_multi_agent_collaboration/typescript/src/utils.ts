/**
 * Multi-agent collaboration utilities — TypeScript port of src/utils.py.
 */

const ARXIV_API_URL = "https://export.arxiv.org/api/query";
const DEFAULT_USER_AGENT = "LF-ADP-Agent/1.0 (mailto:your.email@example.com)";
const DEFAULT_TIMEOUT_MS = 60_000;

export type ToolRecord = Record<string, unknown>;

export function llmError(message: string): ToolRecord[] {
  return [{ error: message }];
}

async function fetchText(url: string, init?: RequestInit): Promise<string> {
  const response = await fetch(url, {
    ...init,
    headers: {
      "User-Agent": DEFAULT_USER_AGENT,
      ...(init?.headers ?? {}),
    },
    signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  return await response.text();
}

function extractTag(source: string, tagName: string): string {
  const match = source.match(
    new RegExp(`<${tagName}[^>]*>([\\s\\S]*?)</${tagName}>`, "i"),
  );
  return match?.[1]?.trim() ?? "";
}

export async function arxivSearchTool(
  query: string,
  maxResults = 5,
): Promise<ToolRecord[]> {
  const url =
    `${ARXIV_API_URL}?search_query=all:${encodeURIComponent(query)}` +
    `&start=0&max_results=${maxResults}`;

  try {
    const xml = await fetchText(url);
    const entries = Array.from(xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)).map(
      (match) => match[1] ?? "",
    );

    return entries.map((entry) => {
      const authors = Array.from(
        entry.matchAll(/<author>\s*<name>([\s\S]*?)<\/name>\s*<\/author>/g),
      ).map((match) => match[1]?.trim() ?? "");

      const pdfLink =
        entry.match(/<link[^>]*title="pdf"[^>]*href="([^"]+)"[^>]*\/?>/i)?.[1] ??
        null;

      return {
        title: extractTag(entry, "title"),
        authors,
        published: extractTag(entry, "published").slice(0, 10),
        url: extractTag(entry, "id"),
        summary: extractTag(entry, "summary"),
        link_pdf: pdfLink,
      };
    });
  } catch (error: unknown) {
    return llmError(error instanceof Error ? error.message : String(error));
  }
}

export async function tavilySearchTool(
  query: string,
  maxResults = 5,
  includeImages = false,
): Promise<ToolRecord[]> {
  try {
    const apiKey = process.env.TAVILY_API_KEY;
    if (!apiKey) {
      throw new Error("TAVILY_API_KEY not found in environment variables.");
    }

    const baseUrl =
      process.env.DLAI_TAVILY_BASE_URL?.trim() || "https://api.tavily.com";
    const response = await fetch(`${baseUrl}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        query,
        max_results: maxResults,
        include_images: includeImages,
      }),
      signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const payload = (await response.json()) as {
      results?: Array<Record<string, unknown>>;
      images?: string[];
    };

    const results: ToolRecord[] = (payload.results ?? []).map((item) => ({
      title: String(item.title ?? ""),
      content: String(item.content ?? ""),
      url: String(item.url ?? ""),
    }));

    if (includeImages) {
      results.push(
        ...(payload.images ?? []).map((imageUrl) => ({ image_url: imageUrl })),
      );
    }

    return results;
  } catch (error: unknown) {
    return llmError(error instanceof Error ? error.message : String(error));
  }
}

export async function wikipediaSearchTool(
  query: string,
  sentences = 5,
): Promise<ToolRecord[]> {
  try {
    const searchUrl =
      "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json" +
      `&srsearch=${encodeURIComponent(query)}`;
    const searchResponse = await fetch(searchUrl, {
      signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
    });

    if (!searchResponse.ok) {
      throw new Error(`HTTP ${searchResponse.status}: ${searchResponse.statusText}`);
    }

    const searchPayload = (await searchResponse.json()) as {
      query?: { search?: Array<{ title?: string }> };
    };

    const title = searchPayload.query?.search?.[0]?.title;
    if (!title) {
      return llmError(`No Wikipedia results for query: ${query}`);
    }

    const summaryUrl =
      "https://en.wikipedia.org/api/rest_v1/page/summary/" +
      encodeURIComponent(title);
    const summaryResponse = await fetch(summaryUrl, {
      signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
    });

    if (!summaryResponse.ok) {
      throw new Error(`HTTP ${summaryResponse.status}: ${summaryResponse.statusText}`);
    }

    const summaryPayload = (await summaryResponse.json()) as {
      title?: string;
      extract?: string;
      content_urls?: {
        desktop?: {
          page?: string;
        };
      };
    };

    const extract = summaryPayload.extract ?? "";
    const clipped = extract
      .split(/(?<=[.!?])\s+/)
      .slice(0, sentences)
      .join(" ");

    return [
      {
        title: summaryPayload.title ?? title,
        summary: clipped || extract,
        url: summaryPayload.content_urls?.desktop?.page ?? "",
      },
    ];
  } catch (error: unknown) {
    return llmError(error instanceof Error ? error.message : String(error));
  }
}
