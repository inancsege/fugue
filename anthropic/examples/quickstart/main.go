// Quickstart: a fugue.Sequential pipeline composed of two different agent
// kinds — a fugue.AgentFunc (no LLM call) followed by an anthropic.Agent.
//
// Run with:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	go run ./examples/quickstart
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/inancsege/fugue"
	"github.com/inancsege/fugue/anthropic"
)

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		log.Fatal("ANTHROPIC_API_KEY is not set")
	}

	// Stage 1: a plain Go function as an Agent. No LLM call — just observes
	// the transcript and emits nothing, so the input passes through unchanged
	// to the next stage. This is what AgentFunc is for: ad-hoc, non-LLM
	// behavior that still composes via the Agent interface.
	logging := fugue.AgentFunc(func(ctx context.Context, in []fugue.Message) ([]fugue.Message, error) {
		fmt.Fprintf(os.Stderr, "[fugue] sending %d message(s) to Claude\n", len(in))
		return nil, nil
	})

	// Stage 2: Anthropic provider adapter. WithSystemPrompt is preferred over
	// injecting a RoleSystem message because Sequential's append-threading
	// would put the system message after the user — Anthropic only accepts
	// system at the top level.
	claude := anthropic.New("claude-sonnet-4-6",
		anthropic.WithSystemPrompt("You are a Go expert. Answer in one sentence."),
		anthropic.WithMaxTokens(200),
	)

	// Compose. logging runs first, then claude. The returned Agent is itself
	// a Runnable — you could nest it inside another Sequential, hand it to
	// AgentAsTool (when it ships), etc.
	pipeline := fugue.Sequential(logging, claude)

	out, err := pipeline.Invoke(context.Background(), []fugue.Message{
		{Role: fugue.RoleUser, Content: []fugue.Part{
			fugue.Text{Text: "Why is io.Reader an interface and not a concrete type?"},
		}},
	})
	if err != nil {
		log.Fatalf("pipeline: %v", err)
	}

	// Sequential returns the full transcript: input + every stage's output.
	// The assistant reply is the final message.
	last := out[len(out)-1]
	for _, p := range last.Content {
		if t, ok := p.(fugue.Text); ok {
			fmt.Println(t.Text)
		}
	}
}
