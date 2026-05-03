package fugue

import (
	"context"
	"iter"
)

// AgentFunc adapts a plain function to Agent — for ad-hoc agents that don't
// need to be a named type. Mirrors the http.HandlerFunc pattern.
//
// Stream lifts Invoke into a single Done=true frame. Implement Agent directly
// when you need real per-token streaming.
type AgentFunc func(ctx context.Context, in []Message) ([]Message, error)

func (f AgentFunc) Invoke(ctx context.Context, in []Message) ([]Message, error) {
	return f(ctx, in)
}

func (f AgentFunc) Stream(ctx context.Context, in []Message) iter.Seq2[Event[[]Message], error] {
	return func(yield func(Event[[]Message], error) bool) {
		out, err := f(ctx, in)
		if err != nil {
			yield(Event[[]Message]{}, err)
			return
		}
		yield(Event[[]Message]{Delta: out, Done: true}, nil)
	}
}
