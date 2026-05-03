package fugue

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

// Compile-time assertion that AgentFunc satisfies Agent.
var _ Agent = AgentFunc(nil)

func TestAgentFunc_InvokeCallsFunction(t *testing.T) {
	want := []Message{msg(RoleAssistant, "out")}
	in := []Message{msg(RoleUser, "hi")}
	var seenIn []Message

	a := AgentFunc(func(ctx context.Context, got []Message) ([]Message, error) {
		seenIn = got
		return want, nil
	})

	got, err := a.Invoke(context.Background(), in)
	if err != nil {
		t.Fatalf("Invoke returned error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	if !reflect.DeepEqual(seenIn, in) {
		t.Errorf("function received %v, want %v", seenIn, in)
	}
}

func TestAgentFunc_StreamYieldsSingleDoneFrame(t *testing.T) {
	want := []Message{msg(RoleAssistant, "out")}
	in := []Message{msg(RoleUser, "hi")}
	var seenIn []Message

	a := AgentFunc(func(ctx context.Context, got []Message) ([]Message, error) {
		seenIn = got
		return want, nil
	})

	var frames []Event[[]Message]
	for ev, err := range a.Stream(context.Background(), in) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		frames = append(frames, ev)
	}

	if len(frames) != 1 {
		t.Fatalf("expected 1 frame, got %d", len(frames))
	}
	if !frames[0].Done {
		t.Errorf("frame should have Done=true")
	}
	if !reflect.DeepEqual(frames[0].Delta, want) {
		t.Errorf("frame.Delta = %v, want %v", frames[0].Delta, want)
	}
	if !reflect.DeepEqual(seenIn, in) {
		t.Errorf("function received %v, want %v", seenIn, in)
	}
}

func TestAgentFunc_StreamYieldsErrorFromFunction(t *testing.T) {
	boom := errors.New("function blew up")
	a := AgentFunc(func(ctx context.Context, _ []Message) ([]Message, error) {
		return nil, boom
	})

	var sawErr error
	frameCount := 0
	for ev, err := range a.Stream(context.Background(), nil) {
		if err != nil {
			sawErr = err
			break
		}
		_ = ev
		frameCount++
	}

	if !errors.Is(sawErr, boom) {
		t.Errorf("errors.Is should find boom, got: %v", sawErr)
	}
	if frameCount != 0 {
		t.Errorf("expected no successful frames before error, got %d", frameCount)
	}
}
