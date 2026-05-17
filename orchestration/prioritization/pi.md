# Pi — Prioritization

**Repository:** https://github.com/earendil-works/pi
**Accessed on:** 2026-05-17
**Source merge:** synthesized from `pi_pi.md`, `pi_codex.md`, and `pi_claude.md`

## Summary

**Only partially implemented.** I did not find a meaningful, general-purpose prioritization system in Pi for deciding which goals, subgoals, or tasks should matter most next.

Pi **does** implement narrower priority mechanisms:

- queue-drain order for pending user messages
- a distinction between steering and follow-up messages
- precedence rules when multiple resources collide

Those are real ordering rules, but they are not the same as strategic prioritization over competing objectives.

## Where it lives

| Concern | Status in Pi |
|---|---|
| Goal/task importance scoring | ❌ none found |
| Queue-drain policy | ✅ `packages/agent/src/types.ts`, `packages/agent/src/agent.ts` |
| Steering-before-follow-up semantics | ✅ `packages/agent/src/agent-loop.ts` |
| Resource precedence ordering | ✅ `packages/coding-agent/src/core/package-manager.ts` |

## Key code excerpts

Source: `packages/agent/src/types.ts:38-44`

```ts
/**
 * Controls how many queued user messages are injected when the agent loop reaches a queue drain point.
 *
 * - "all": drain and inject every queued message at that point.
 * - "one-at-a-time": drain and inject only the oldest queued message, leaving the rest queued for later drain points.
 */
export type QueueMode = "all" | "one-at-a-time";
```

Why this matters: Pi does have an explicit notion of ordering policy for queued work, but it is a delivery rule, not a semantic priority model.

Source: `packages/agent/src/agent.ts:118-144`

```ts
class PendingMessageQueue {
	private messages: AgentMessage[] = [];
	...
	drain(): AgentMessage[] {
		if (this.mode === "all") {
			const drained = this.messages.slice();
			this.messages = [];
			return drained;
		}

		const first = this.messages[0];
		...
		return [first];
	}
}
```

Why this matters: Pi can prioritize the oldest pending message by queue policy, but it does not score messages by value, urgency, or dependency.

Source: `packages/agent/src/agent.ts:260-267` and `packages/agent/src/agent-loop.ts:166-178,253-261`

```ts
/** Queue a message to be injected after the current assistant turn finishes. */
steer(message: AgentMessage): void {
	this.steeringQueue.enqueue(message);
}

/** Queue a message to run only after the agent would otherwise stop. */
followUp(message: AgentMessage): void {
	this.followUpQueue.enqueue(message);
}
```

```ts
// Check for steering messages at start (user may have typed while waiting)
let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];
...
// Agent would stop here. Check for follow-up messages.
const followUpMessages = (await config.getFollowUpMessages?.()) || [];
if (followUpMessages.length > 0) {
	pendingMessages = followUpMessages;
	continue;
}
```

Why this matters: Pi clearly prioritizes steering messages over follow-up messages. Again, that is execution order, not strategic prioritization.

Source: `packages/coding-agent/src/core/package-manager.ts:162-176`

```ts
/**
 * Compute a numeric precedence rank for a resource based on its metadata.
 * Lower rank = higher precedence.
 *
 * Precedence (highest to lowest):
 *   0  project + settings entry
 *   1  project + auto-discovered
 *   2  user + settings entry
 *   3  user + auto-discovered
 *   4  package resource
 */
function resourcePrecedenceRank(m: PathMetadata): number {
```

Why this matters: Pi does implement deterministic precedence rules, but for configuration/resource resolution rather than task prioritization.

## Tradeoffs and limitations

- Pi favors explicit, predictable ordering semantics over opaque prioritization heuristics.
- That keeps behavior easier to reason about.
- The downside is that Pi does not help decide which objective is most important when there are multiple competing goals.

## Final word

Pi has **ordering and precedence rules**, but not a meaningful first-class prioritization system for goals or tasks.
