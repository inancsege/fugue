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
//
// Stream contract: implementations MUST emit at least one frame, and the
// final emitted frame MUST have Done=true (for the success case). Combinators
// rely on the terminal Done frame to know when to commit downstream state.
// On error, implementations yield (Event[O]{}, err) and stop — no further
// frames; the consumer MUST ignore Delta when err is non-nil.
type Runnable[I, O any] interface {
	Invoke(ctx context.Context, in I) (O, error)
	Stream(ctx context.Context, in I) iter.Seq2[Event[O], error]
}

// Event is a single frame emitted by a streaming [Runnable].
//
// Delta carries the partial output for this frame; chunking semantics are
// provider-defined. Done is true on the final frame.
//
// For Runnables shaped Runnable[_, []Message] (i.e. [Agent]), Delta is
// cumulative: each frame's Delta holds the stage's complete output as of
// that frame, not a per-token diff. Combinators that thread output between
// stages (notably Sequential.Stream) rely on this. Provider adapters that
// emit per-token diffs MUST accumulate before yielding.
//
// When the iterator yields a non-nil error, Delta MUST be ignored — the zero
// value is used and carries no meaning.
type Event[O any] struct {
	Delta O
	Done  bool
}

// Agent is the canonical alias for message-shaped Runnables.
//
// All public combinators (Sequential, Parallel, Router) take and return Agent.
// Use the generic [Runnable] form for typed conversion components at the
// edges of a flow.
type Agent = Runnable[[]Message, []Message]
