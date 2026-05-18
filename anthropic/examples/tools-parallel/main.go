// Command tools-parallel demonstrates fugue's parallel tool dispatch. The
// model is given two tools (search and weather) and a prompt that requires
// both. fugue runs the tools concurrently when the model emits multiple
// tool_use blocks in a single turn.
//
// Set ANTHROPIC_API_KEY and run:
//
//	go run ./examples/tools-parallel
//
// From the anthropic/ directory.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/inancsege/fugue"
	"github.com/inancsege/fugue/anthropic"
)

type SearchIn struct {
	Query string `json:"query" fugue:"search query"`
}
type SearchOut struct {
	Hits []string `json:"hits"`
}

func search(_ context.Context, in SearchIn) (SearchOut, error) {
	// Simulated latency to make parallel dispatch visible.
	time.Sleep(200 * time.Millisecond)
	return SearchOut{Hits: []string{"result for " + in.Query}}, nil
}

type WeatherIn struct {
	City string `json:"city" fugue:"city name"`
}
type WeatherOut struct {
	TempF int `json:"temp_f"`
}

func weather(_ context.Context, in WeatherIn) (WeatherOut, error) {
	time.Sleep(200 * time.Millisecond)
	if in.City == "" {
		return WeatherOut{}, fmt.Errorf("city is required")
	}
	return WeatherOut{TempF: 72}, nil
}

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		log.Fatal("ANTHROPIC_API_KEY is required")
	}

	agent := anthropic.New("claude-sonnet-4-6",
		anthropic.WithSystemPrompt("Use both tools when the user's question needs both. Be brief."),
		anthropic.WithTools(
			fugue.Tool("search", "Search the corpus for matching documents.", search),
			fugue.Tool("get_weather", "Get the current weather for a city.", weather),
		),
		anthropic.WithMaxSteps(4),
	)

	start := time.Now()
	out, err := agent.Invoke(context.Background(), []fugue.Message{{
		Role:    fugue.RoleUser,
		Content: []fugue.Part{fugue.Text{Text: "Search for 'go generics' AND tell me the weather in NYC."}},
	}})
	if err != nil {
		log.Fatalf("Invoke: %v", err)
	}
	fmt.Printf("(took %v)\n\n", time.Since(start))

	final := out[len(out)-1]
	for _, p := range final.Content {
		if t, ok := p.(fugue.Text); ok {
			fmt.Println(t.Text)
		}
	}
}
