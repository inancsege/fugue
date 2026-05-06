package fugue

import (
	"errors"
	"fmt"
)

// ErrNoRoute is the sentinel that wraps a Router's "decide returned a key
// absent from the routes map" condition. It is exposed inside the *RouteError
// returned by Router so callers can write errors.Is(err, fugue.ErrNoRoute) to
// distinguish unknown-key from decide-itself-failed.
var ErrNoRoute = errors.New("no route")

// StageError reports which stage of a combinator failed.
//
// Returned by Sequential (and later Parallel/Router) when an underlying agent
// fails. Use errors.As to recover the index and any partial transcript
// produced before the failing stage ran.
type StageError struct {
	Index   int       // 0-based position in the combinator
	Err     error     // the underlying error
	Partial []Message // transcript accumulated before the failing stage ran (may be nil)
}

func (e *StageError) Error() string {
	return fmt.Sprintf("stage %d: %v", e.Index, e.Err)
}

func (e *StageError) Unwrap() error { return e.Err }

// RouteError reports a routing-layer failure from a Router combinator.
//
// Distinct from *StageError (which is for ordered combinator stages). Routing
// is dispatch, not a numbered stage — the failing key is more useful than an
// index.
//
// Returned by Router when:
//   - decide itself returns an error (Key is empty)
//   - decide returns a key not present in routes (Key is the unknown key)
//
// When the chosen agent itself fails, that error passes through bare — it is
// not wrapped in *RouteError.
type RouteError struct {
	Key string // the key returned by decide (empty if decide itself errored)
	Err error  // the underlying error
}

func (e *RouteError) Error() string {
	if e.Key == "" {
		return fmt.Sprintf("route: %s", e.Err.Error())
	}
	return fmt.Sprintf("route %q: %s", e.Key, e.Err.Error())
}

func (e *RouteError) Unwrap() error { return e.Err }
