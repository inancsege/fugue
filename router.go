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
// is wrapped in *RouteError. If the chosen agent itself errors, that error
// passes through bare.
//
// Stream passes through the chosen agent's stream verbatim — full provider-
// level token streaming through Router.
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
	return func(yield func(Event[[]Message], error) bool) {}
}

// errNoRoute returns the sentinel-style error used inside *RouteError when
// decide returned a key that doesn't exist in the routes map.
func errNoRoute(key string) error {
	return fmt.Errorf("no route for %q", key)
}
