package fugue

import (
	"context"
	"iter"
	"slices"
	"sync"
)

// Parallel runs agents concurrently against the same input.
//
// Each agent receives a clone of the input messages. The returned Agent's
// output is each agent's output concatenated in agent-index order — no input
// prefix. Sequential is the framework's transcript-builder; Parallel is a
// payload-producer. This lets Sequential(planner, Parallel(...)) compose
// without duplicating the input.
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
	outs := make([][]Message, len(p.agents))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	errs := make(chan error, len(p.agents))
	var wg sync.WaitGroup
	wg.Add(len(p.agents))
	for i, a := range p.agents {
		i, a := i, a
		go func() {
			defer wg.Done()
			inClone := slices.Clone(in)
			out, err := a.Invoke(ctx, inClone)
			if err != nil {
				errs <- &StageError{Index: i, Err: err, Partial: inClone}
				cancel()
				return
			}
			outs[i] = out
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			return nil, err
		}
	}

	// Return only the agents' outputs concatenated in index order — no input
	// prefix. Sequential is the transcript-builder; Parallel is a payload-
	// producer. This lets Sequential(planner, Parallel(...)) compose without
	// duplicating the input.
	var collected []Message
	for _, out := range outs {
		collected = append(collected, out...)
	}
	return collected, nil
}

func (p *parallel) Stream(ctx context.Context, in []Message) iter.Seq2[Event[[]Message], error] {
	return func(yield func(Event[[]Message], error) bool) {
		out, err := p.Invoke(ctx, in)
		if err != nil {
			yield(Event[[]Message]{}, err)
			return
		}
		yield(Event[[]Message]{Delta: out, Done: true}, nil)
	}
}
