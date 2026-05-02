package fugue

import (
	"context"
	"iter"
	"slices"
)

// Sequential runs agents in order, threading the conversation forward.
//
// Each agent receives the input messages plus everything every prior agent
// emitted, concatenated in order. The returned Agent's output is the final
// transcript: input + every agent's output, in order.
//
// Sequential() panics — a zero-agent pipeline is a programming bug.
// Sequential(a) returns a directly.
func Sequential(agents ...Agent) Agent {
	if len(agents) == 0 {
		panic("fugue: Sequential() requires at least one agent")
	}
	if len(agents) == 1 {
		return agents[0]
	}
	return &sequential{agents: slices.Clone(agents)}
}

type sequential struct {
	agents []Agent
}

func (s *sequential) Invoke(ctx context.Context, in []Message) ([]Message, error) {
	transcript := slices.Clone(in)
	for i, a := range s.agents {
		out, err := a.Invoke(ctx, transcript)
		if err != nil {
			return nil, &StageError{Index: i, Err: err, Partial: transcript}
		}
		transcript = append(transcript, out...)
	}
	return transcript, nil
}

func (s *sequential) Stream(ctx context.Context, in []Message) iter.Seq2[Event[[]Message], error] {
	return func(yield func(Event[[]Message], error) bool) {
		transcript := slices.Clone(in)
		last := len(s.agents) - 1

		for i, a := range s.agents {
			var stageOut []Message
			isLast := i == last

			for ev, err := range a.Stream(ctx, transcript) {
				if err != nil {
					yield(Event[[]Message]{}, &StageError{Index: i, Err: err, Partial: transcript})
					return
				}
				if ev.Done && !isLast {
					ev.Done = false
				}
				stageOut = ev.Delta
				if !yield(ev, nil) {
					return
				}
			}
			transcript = append(transcript, stageOut...)
		}
	}
}
