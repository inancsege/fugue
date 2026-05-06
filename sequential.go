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
		// Clip before append so a misbehaving agent that resliced its input
		// past len cannot observe writes from subsequent stages through the
		// shared backing array.
		transcript = append(slices.Clip(transcript), out...)
	}
	return transcript, nil
}

func (s *sequential) Stream(ctx context.Context, in []Message) iter.Seq2[Event[[]Message], error] {
	return func(yield func(Event[[]Message], error) bool) {
		transcript := slices.Clone(in)
		last := len(s.agents) - 1

		for i, a := range s.agents {
			var stageOut []Message
			sawDone := false
			isLast := i == last

			for ev, err := range a.Stream(ctx, transcript) {
				if err != nil {
					yield(Event[[]Message]{}, &StageError{Index: i, Err: err, Partial: transcript})
					return
				}
				if ev.Done {
					sawDone = true
					if !isLast {
						ev.Done = false
					}
				}
				// Event.Delta is cumulative for []Message Runnables — see runnable.go
				// godoc. Each frame's Delta holds the stage's complete output so far.
				stageOut = ev.Delta
				if !yield(ev, nil) {
					return
				}
			}
			// Defensive: a stage that violated the Runnable.Stream contract by
			// emitting zero frames or no terminal Done frame would otherwise
			// leave the consumer waiting. On the last stage we synthesise the
			// terminal frame so downstream consumers see the Done they're
			// promised. Mid-pipeline we just thread whatever stageOut we have.
			if isLast && !sawDone {
				if !yield(Event[[]Message]{Delta: stageOut, Done: true}, nil) {
					return
				}
			}
			transcript = append(slices.Clip(transcript), stageOut...)
		}
	}
}
