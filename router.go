package fugue

import (
	"context"
	"fmt"
	"iter"
)

// RouteFunc is the decision function for a Router. It receives the same
// context and input the chosen agent would receive, and returns the key
// of the route to dispatch to.
type RouteFunc func(ctx context.Context, in []Message) (string, error)

// Router dispatches input to exactly one of the named agents based on the
// decide function. The chosen agent's output is returned unchanged (no input
// prefix — Sequential is the transcript-builder).
//
// Router(nil, ...) and Router(_, nil/empty map) panic — both are programming bugs.
//
// If decide returns an error, or returns a key absent from routes, the error
// is wrapped in *RouteError. The unknown-key case wraps [ErrNoRoute], so
// callers can write errors.Is(err, fugue.ErrNoRoute) to distinguish it from
// a decide-itself-failed error. If the chosen agent itself errors, that
// error passes through bare.
//
// RouteFunc returns only (key, error) — it cannot transform the input passed
// to the chosen agent. If you need that, wrap the chosen agents in
// [AgentFunc] adapters that do the transformation.
//
// Stream passes through the chosen agent's stream verbatim — full provider-
// level token streaming through Router. decide runs lazily on the consumer's
// first iteration of the returned sequence, not at Stream call time.
func Router(decide RouteFunc, routes map[string]Agent) Agent {
	if decide == nil {
		panic("fugue: Router() requires a non-nil decide function")
	}
	if len(routes) == 0 {
		panic("fugue: Router() requires at least one route")
	}
	return &router{decide: decide, routes: routes}
}

type router struct {
	decide RouteFunc
	routes map[string]Agent
}

func (r *router) Invoke(ctx context.Context, in []Message) ([]Message, error) {
	key, err := r.decide(ctx, in)
	if err != nil {
		return nil, &RouteError{Key: "", Err: err}
	}
	chosen, ok := r.routes[key]
	if !ok {
		return nil, &RouteError{Key: key, Err: errNoRoute(key)}
	}
	return chosen.Invoke(ctx, in)
}

func (r *router) Stream(ctx context.Context, in []Message) iter.Seq2[Event[[]Message], error] {
	return func(yield func(Event[[]Message], error) bool) {
		key, err := r.decide(ctx, in)
		if err != nil {
			yield(Event[[]Message]{}, &RouteError{Key: "", Err: err})
			return
		}
		chosen, ok := r.routes[key]
		if !ok {
			yield(Event[[]Message]{}, &RouteError{Key: key, Err: errNoRoute(key)})
			return
		}
		for ev, err := range chosen.Stream(ctx, in) {
			if !yield(ev, err) {
				return
			}
		}
	}
}

// errNoRoute returns the error used inside *RouteError when decide returned
// a key that doesn't exist in the routes map. It wraps the exported
// [ErrNoRoute] sentinel so callers can use errors.Is to distinguish this
// case from a decide-itself-failed RouteError.
func errNoRoute(key string) error {
	return fmt.Errorf("%w for %q", ErrNoRoute, key)
}
