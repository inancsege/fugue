package fugue

import "fmt"

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
	return fmt.Sprintf("stage %d: %s", e.Index, e.Err.Error())
}

func (e *StageError) Unwrap() error { return e.Err }
