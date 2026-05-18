// Command tools-basic demonstrates fugue's typed-tools API with a single
// tool. Set ANTHROPIC_API_KEY and run:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	go run ./examples/tools-basic
//
// From the anthropic/ directory.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/inancsege/fugue"
	"github.com/inancsege/fugue/anthropic"
)

// WeatherIn is the tool's typed input. Struct tags drive schema reflection.
type WeatherIn struct {
	City string `json:"city" fugue:"city name to look up weather for"`
}

// WeatherOut is the tool's typed output. JSON-marshaled and shown to the model.
type WeatherOut struct {
	TempF       int    `json:"temp_f"`
	Description string `json:"description"`
}

func weather(_ context.Context, in WeatherIn) (WeatherOut, error) {
	// In a real tool, call a weather API. Here, hardcoded for the example.
	if in.City == "" {
		return WeatherOut{}, fmt.Errorf("city is required")
	}
	return WeatherOut{TempF: 72, Description: "sunny"}, nil
}

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		log.Fatal("ANTHROPIC_API_KEY is not set")
	}

	agent := anthropic.New("claude-sonnet-4-6",
		anthropic.WithSystemPrompt("Answer using the weather tool. Be brief."),
		anthropic.WithTools(fugue.Tool("get_weather", "Get the current weather for a city.", weather)),
		anthropic.WithMaxSteps(4),
	)

	out, err := agent.Invoke(context.Background(), []fugue.Message{{
		Role:    fugue.RoleUser,
		Content: []fugue.Part{fugue.Text{Text: "What's the weather in NYC?"}},
	}})
	if err != nil {
		log.Fatalf("Invoke: %v", err)
	}

	// The final assistant message is the last one in the trace.
	final := out[len(out)-1]
	for _, p := range final.Content {
		if t, ok := p.(fugue.Text); ok {
			fmt.Println(t.Text)
		}
	}
}
