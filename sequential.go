package fugue

import (
	"context"
	"iter"
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
	return &sequential{agents: agents}
}

type sequential struct {
	agents []Agent
}

func (s *sequential) Invoke(ctx context.Context, in []Message) ([]Message, error) {
	return nil, nil
}

func (s *sequential) Stream(ctx context.Context, in []Message) iter.Seq2[Event[[]Message], error] {
	return func(yield func(Event[[]Message], error) bool) {}
}
