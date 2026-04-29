package fugue

import (
	"context"
	"iter"
)

// Runnable is the unifying interface implemented by agents and combinators.
//
// Both methods carry the same semantics; Stream emits incremental frames where
// natural and a single terminal frame otherwise. Components that have nothing
// useful to stream may implement Stream by lifting the result of Invoke into a
// single-frame iterator.
type Runnable[I, O any] interface {
	Invoke(ctx context.Context, in I) (O, error)
	Stream(ctx context.Context, in I) iter.Seq2[Event[O], error]
}

// Event is a single frame emitted by a streaming [Runnable].
//
// Delta carries the partial output for this frame; chunking semantics are
// provider-defined. Done is true on the final frame.
type Event[O any] struct {
	Delta O
	Done  bool
}

// Agent is the canonical alias for message-shaped Runnables.
//
// All public combinators (Sequential, Parallel, Router, AgentAsTool) take and
// return Agent. Use the generic [Runnable] form for typed conversion components
// at the edges of a flow.
type Agent = Runnable[[]Message, []Message]
