package fugue

import (
	"context"
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
