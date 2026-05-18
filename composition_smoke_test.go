package fugue

import (
	"context"
	"iter"
	"reflect"
	"testing"
)

// Smoke test for cross-combinator composition. Catches the input-duplication
// regression that prompted the Parallel output-shape fix.
func TestComposition_SequentialOfParallel(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "out-a")}}
	b := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "out-b")}}
	c := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "out-c")}}

	in := []Message{msg(RoleUser, "hi")}
	got, err := Sequential(a, Parallel(b, c)).Invoke(context.Background(), in)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	want := []Message{
		msg(RoleUser, "hi"),
		msg(RoleAssistant, "out-a"),
		msg(RoleAssistant, "out-b"),
		msg(RoleAssistant, "out-c"),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Sequential(a, Parallel(b, c)) duplicated input\n got: %v\nwant: %v", got, want)
	}
}

func TestComposition_SequentialOfRouter(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "answer-a")}}
	b := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "answer-b")}}

	classifier := agentFunc(func(ctx context.Context, in []Message) ([]Message, error) {
		return []Message{msg(RoleAssistant, "classified-as-a")}, nil
	})
	decide := func(ctx context.Context, in []Message) (string, error) {
		// A real router would inspect the classifier's output (last message).
		return "a", nil
	}

	in := []Message{msg(RoleUser, "hi")}
	got, err := Sequential(classifier, Router(decide, map[string]Agent{"a": a, "b": b})).Invoke(context.Background(), in)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	want := []Message{
		msg(RoleUser, "hi"),
		msg(RoleAssistant, "classified-as-a"),
		msg(RoleAssistant, "answer-a"),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Sequential(classifier, Router(...)) composition mismatch\n got: %v\nwant: %v", got, want)
	}
}

// stubToolAgent is a fake fugue.Agent that returns a fixed trace.
// Mimics a tool-using agent (multi-message output) without needing a real
// Anthropic adapter. Used to pin that Sequential/Router compose with the
// multi-message shape produced by invokeWithTools.
type stubToolAgent struct {
	trace []Message
}

func (s *stubToolAgent) Invoke(_ context.Context, _ []Message) ([]Message, error) {
	return s.trace, nil
}

func (s *stubToolAgent) Stream(_ context.Context, _ []Message) iter.Seq2[Event[[]Message], error] {
	return func(yield func(Event[[]Message], error) bool) {
		yield(Event[[]Message]{Delta: s.trace, Done: true}, nil)
	}
}

func TestComposition_SequentialOfToolUsingAgent(t *testing.T) {
	toolAgent := &stubToolAgent{trace: []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "u1", Name: "f", Arguments: []byte(`{}`)}}},
		{Role: RoleTool, ToolCallID: "u1", Content: []Part{Text{Text: "result"}}},
		{Role: RoleAssistant, Content: []Part{Text{Text: "final"}}},
	}}
	// downstream "formatter" prefixes the last text with "OUT:".
	formatter := AgentFunc(func(_ context.Context, in []Message) ([]Message, error) {
		last := in[len(in)-1]
		var text string
		if t, ok := last.Content[0].(Text); ok {
			text = t.Text
		}
		return []Message{{Role: RoleAssistant, Content: []Part{Text{Text: "OUT:" + text}}}}, nil
	})

	pipeline := Sequential(toolAgent, formatter)
	out, err := pipeline.Invoke(context.Background(), []Message{{Role: RoleUser, Content: []Part{Text{Text: "go"}}}})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	// Sequential returns input + every stage's output. Find the formatter output.
	found := false
	for _, m := range out {
		if len(m.Content) > 0 {
			if t, ok := m.Content[0].(Text); ok && t.Text == "OUT:final" {
				found = true
				break
			}
		}
	}
	if !found {
		t.Errorf("expected formatter output OUT:final in trace, got %+v", out)
	}
}

func TestComposition_RouterOfToolUsingAgent(t *testing.T) {
	toolAgent := &stubToolAgent{trace: []Message{
		{Role: RoleAssistant, Content: []Part{Text{Text: "tool-using answer"}}},
	}}
	plain := AgentFunc(func(_ context.Context, _ []Message) ([]Message, error) {
		return []Message{{Role: RoleAssistant, Content: []Part{Text{Text: "plain answer"}}}}, nil
	})
	r := Router(
		func(_ context.Context, _ []Message) (string, error) { return "tool", nil },
		map[string]Agent{"tool": toolAgent, "plain": plain},
	)
	out, err := r.Invoke(context.Background(), []Message{{Role: RoleUser, Content: []Part{Text{Text: "x"}}}})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 message in out, got %d", len(out))
	}
	if txt, ok := out[0].Content[0].(Text); !ok || txt.Text != "tool-using answer" {
		t.Errorf("router did not forward tool-using agent's trace verbatim: %+v", out)
	}
}
