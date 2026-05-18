# fugue

Multi-voice agent orchestration for Go. Code-first composition — no YAML, no Markdown, just Go.

> **Status:** v0, operational with Anthropic. The core API is stable. Some pieces are still missing — the agent-as-tool primitive, more providers — but you can wire real multi-agent pipelines today.

## Why fugue

Most multi-agent frameworks define agents in YAML or Markdown. fugue defines them in Go.

That trade buys real type safety, real refactoring, real testing, and a real debugger — in
exchange for a few extra lines at construction time. Orchestration composes like the rest of
your code, not like a config file.

## Quickstart

```go
import (
    "context"
    "github.com/inancsege/fugue"
    "github.com/inancsege/fugue/anthropic"
)

planner := anthropic.New("claude-sonnet-4-6",
    anthropic.WithSystemPrompt("Plan, don't execute. One sentence per step."))
executor := anthropic.New("claude-sonnet-4-6",
    anthropic.WithSystemPrompt("Execute the plan. Return only the result."))

pipeline := fugue.Sequential(planner, executor)

out, err := pipeline.Invoke(ctx, []fugue.Message{{
    Role: fugue.RoleUser,
    Content: []fugue.Part{fugue.Text{Text: "Migrate the users table to UUIDs"}},
}})
```

Set `ANTHROPIC_API_KEY` and run a working end-to-end example with `go run ./examples/quickstart`
from the `anthropic/` directory.

## Design

A few decisions are locked. They shape everything that follows.

- **Combinators first.** `Sequential`, `Parallel`, and `Router` are the headline API. Most
  real workflows are sequential / parallel / routing, and Go's `io` and `errgroup` show how
  far small composable operators can take you.
- **Agent-as-tool** is a supported primitive for swarm-style delegation when combinators
  aren't enough. _Not yet shipped._
- **No graph builder in v1.** Outer iteration is a `for` loop in your code. Inner iteration
  is the agent's `MaxSteps`. That's the entire point of being code-first.
- **One `Runnable[I, O]` interface** for agents and combinators alike, carrying both `Invoke`
  and `Stream`. Streaming uses Go 1.23's `iter.Seq2`.
- **Typed tools.** Tool input/output schemas come from Go struct tags via reflection — not
  hand-written JSON schema, not a YAML descriptor.

## Combinators

### Sequential — chain agents, thread the conversation

```go
pipeline := fugue.Sequential(planner, critic, executor)
```

Each stage receives `input + every prior stage's output`. Returns the full transcript
(`input + out_0 + out_1 + ...`). This is the framework's transcript-builder.

### Parallel — fan out, collect outputs

```go
ensemble := fugue.Parallel(claude1, claude2, claude3)
```

All agents run concurrently against the same input. Returns `out_0 + out_1 + out_2` in
agent-index order (deterministic regardless of completion timing). **No input prefix** —
Sequential builds the transcript, Parallel produces the payload. Fail-fast: the first
non-nil error cancels sibling contexts.

### Router — dispatch to one agent based on input

```go
router := fugue.Router(
    func(ctx context.Context, in []fugue.Message) (string, error) {
        if isCodeQuestion(in) {
            return "code", nil
        }
        return "writing", nil
    },
    map[string]fugue.Agent{
        "code":    coder,
        "writing": writer,
    },
)
```

Returns the chosen agent's output verbatim. `Stream` passes provider tokens through
unchanged — full per-token streaming through the router. Routing-layer failures
(decide errors, unknown keys) are wrapped in `*RouteError`; the chosen agent's errors
pass through bare.

### Composing them

```go
fugue.Sequential(
    classifier,
    fugue.Router(decide, map[string]fugue.Agent{
        "code":    coder,
        "math":    fugue.Parallel(solver1, solver2),
        "writing": writer,
    }),
    formatter,
)
```

## AgentFunc — wrap a plain Go function as an Agent

For ad-hoc stages that don't need to be a named type:

```go
logging := fugue.AgentFunc(func(ctx context.Context, in []fugue.Message) ([]fugue.Message, error) {
    log.Printf("sending %d msgs to next stage", len(in))
    return nil, nil // emit nothing; transcript passes through unchanged
})

pipeline := fugue.Sequential(logging, claude)
```

Mirrors `http.HandlerFunc`. `Stream` lifts `Invoke` into a single `Done=true` frame; implement
the `Agent` interface directly when you need real per-token streaming.

## Tools

Define a tool as a plain Go function with a struct input. fugue reflects the
struct into a JSON Schema and threads `(args → fn → result)` through the
provider's tool-use loop.

```go
type SearchIn struct {
    Query string `json:"query" fugue:"the search query"`
    Limit int    `json:"limit,omitempty" fugue:"max results, default 10"`
}
type SearchOut struct {
    Hits []string `json:"hits"`
}

func search(ctx context.Context, in SearchIn) (SearchOut, error) {
    // ...
    return SearchOut{Hits: []string{"first"}}, nil
}

agent := anthropic.New("claude-sonnet-4-6",
    anthropic.WithSystemPrompt("Answer with the search tool."),
    anthropic.WithTools(fugue.Tool("search", "Search the corpus", search)),
    anthropic.WithMaxSteps(8),
)

out, err := agent.Invoke(ctx, []fugue.Message{{
    Role: fugue.RoleUser, Content: []fugue.Part{fugue.Text{Text: "find: hello"}},
}})
```

`Invoke` returns the **full trace**: each assistant turn (including
`tool_use` blocks) and each `tool_result` in order, ending with the final
text turn. `Sequential` threads the trace into the next stage.

### Schema reflection

| Go type | JSON Schema |
|---|---|
| `string`, `bool` | `string`, `boolean` |
| `int*`, `uint*` | `integer` |
| `float32`, `float64` | `number` |
| `[]T`, `map[string]T` | `array` / `object` (`additionalProperties`) |
| nested struct | `object` |
| `*T` | same as `T`, not required |
| `json.RawMessage` | `{}` (any JSON) |

Tags:
- `json:"name,omitempty"` — wire name; `omitempty` (or `*T`) marks not-required.
- `fugue:"..."` — property description shown to the model.
- `fugueEnum:"a,b,c"` — string enum (only on `string` fields).

Unsupported types (channels, functions, `any`, `time.Time`, recursive structs,
non-string map keys) panic at `fugue.Tool(...)` construction with a message
including the field path — these are programming bugs, not runtime errors.

### Errors and the loop

- A tool fn returning `error` is *not* fatal. fugue feeds the error back to
  the model as `tool_result{is_error: true}` so it can retry, route, or
  apologize. Tool panics are recovered with the same treatment.
- Programming-bug errors (e.g. an `Out` type that can't be JSON-marshaled)
  bubble out of `Invoke` as transport errors.
- If the model never stops requesting tools, the loop terminates with
  `*fugue.ToolLoopError{Steps: N}` after `WithMaxSteps` turns (default 8).

### Raw escape hatch

For recursive schemas, provider-specific JSON Schema features, or non-struct
outputs, use `fugue.RawTool(name, desc, schema, fn)` — same loop, you write
the schema yourself.

## Streaming

Every `Runnable` has a `Stream` method returning a Go 1.23 `iter.Seq2` of
`(Event[O], error)`. The simplest consumer is a `range` loop:

```go
for ev, err := range pipeline.Stream(ctx, in) {
    if err != nil {
        return err
    }
    if ev.Done {
        // ev.Delta is the cumulative final output.
        return commit(ev.Delta)
    }
    render(ev.Delta) // cumulative — replace, don't append
}
```

`Event.Delta` is **cumulative** for `[]Message`-shaped Runnables: each frame
holds the stage's complete output as of that frame, not a per-token diff. The
final frame has `Done == true`. When `err` is non-nil, ignore `Delta`.

For mid-stream cancel — e.g. abort as soon as a stop word appears — use
`iter.Pull2` to drive the iterator manually; calling the returned `stop` is
how you signal the producer to release its connection:

```go
next, stop := iter.Pull2(pipeline.Stream(ctx, in))
defer stop()
for {
    ev, err, ok := next()
    if !ok {
        return nil
    }
    if err != nil {
        return err
    }
    if shouldStop(ev.Delta) {
        return nil // defer stop() releases the upstream stream
    }
}
```

The combinators preserve streaming contracts in different ways:

- **Sequential** forwards each stage's frames inline, flipping inner stages'
  `Done` bits to false so only the very last frame of the final stage carries
  `Done=true`.
- **Router** forwards the chosen agent's frames verbatim — full per-token
  streaming through the router. `decide` runs lazily on the consumer's first
  iteration.
- **Parallel** is buffered: it runs `Invoke` then emits one terminal frame.
  Frames do not interleave (the order-preserving contract makes that
  impossible without per-frame agent identification).

## Providers

| Module | Status |
|---|---|
| [`github.com/inancsege/fugue/anthropic`](./anthropic) | Messages API: text, image, tool_use, tool_result, thinking blocks. Real per-token streaming via SSE. |

The core module has zero LLM dependencies. Each provider is a nested module — depend only on
the providers you actually use.

## Errors

- **`*StageError`** for ordered-combinator failures (`Sequential`, `Parallel`). Carries the
  failing stage's index and partial state.
- **`*RouteError`** for routing-layer failures (`Router` — `decide` errored, or returned an
  unknown key). Carries the failing key.
- **`*ToolLoopError`** when a tool-use loop exceeds its step budget. Carries the
  step count.

All three work with `errors.As` to recover their typed values; `*StageError` and `*RouteError` also implement `Unwrap` (`*ToolLoopError` does not — there is no underlying cause).

## License

[MIT](LICENSE).
