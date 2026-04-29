# fugue

Multi-voice agent orchestration for Go. Code-first composition — no YAML, no Markdown, just Go.

> **Status:** early scaffolding. The API is being designed in the open. Not yet ready for use.

## Why fugue

Most multi-agent frameworks define agents in YAML or Markdown. fugue defines them in Go.

That trade buys real type safety, real refactoring, real testing, and a real debugger — in
exchange for a few extra lines at construction time. Orchestration composes like the rest of
your code, not like a config file.

## Design

A few decisions are locked. They shape everything that follows.

- **Combinators first.** `Sequential`, `Parallel`, and `Router` are the headline API. Most
  real workflows are sequential / parallel / routing, and Go's `io` and `errgroup` show how
  far small composable operators can take you.
- **Agent-as-tool** is a supported primitive for swarm-style delegation when combinators
  aren't enough.
- **No graph builder in v1.** Outer iteration is a `for` loop in your code. Inner iteration
  is the agent's `MaxSteps`. That's the entire point of being code-first.
- **One `Runnable[I, O]` interface** for agents and combinators alike, carrying both `Invoke`
  and `Stream`. Streaming uses Go 1.23's `iter.Seq2`.
- **Typed tools.** Tool input/output schemas come from Go struct tags via reflection — not
  hand-written JSON schema, not a YAML descriptor.

## Status

This repository currently contains the core types: `Runnable`, `Message`, `Part`, `Event`,
and the `Agent` alias. Up next: provider adapters (Anthropic, OpenAI), the combinator
package (`Sequential`, `Parallel`, `Router`, `AgentAsTool`), and the typed tool layer.

## License

[MIT](LICENSE).
