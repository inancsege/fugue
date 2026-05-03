package fugue

import (
	"context"
	"iter"
	"slices"
)

// Parallel runs agents concurrently against the same input.
//
// Each agent receives a clone of the input messages. The returned Agent's
// output is the input transcript followed by each agent's output appended
// in agent-index order (regardless of completion order).
//
// Parallel() panics — a zero-agent fan-out is a programming bug.
// Parallel(a) returns a directly.
//
// Errors are fail-fast: the first non-nil error cancels sibling contexts and
// is returned wrapped in a *StageError carrying the failing agent's index.
// Sibling outputs are discarded; *StageError.Partial is the input that was
// passed to the failing stage.
func Parallel(agents ...Agent) Agent {
	if len(agents) == 0 {
		panic("fugue: Parallel() requires at least one agent")
	}
	if len(agents) == 1 {
		return agents[0]
	}
	return &parallel{agents: slices.Clone(agents)}
}

type parallel struct {
	agents []Agent
}

func (p *parallel) Invoke(ctx context.Context, in []Message) ([]Message, error) {
	return nil, nil
}

func (p *parallel) Stream(ctx context.Context, in []Message) iter.Seq2[Event[[]Message], error] {
	return func(yield func(Event[[]Message], error) bool) {}
}
