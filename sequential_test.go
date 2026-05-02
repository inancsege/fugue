package fugue

import (
	"context"
	"errors"
	"iter"
	"reflect"
	"testing"
)

// fakeAgent is a controllable Agent for tests. Implements Runnable[[]Message, []Message].
//
// Invoke returns invokeOut and invokeErr (in that order — err overrides out).
// Stream emits each Event in streamFrames in order, then yields streamErr if non-nil.
// seenInvokeIn / seenStreamIn capture the input each method received, for assertions.
type fakeAgent struct {
	invokeOut    []Message
	invokeErr    error
	streamFrames []Event[[]Message]
	streamErr    error
	seenInvokeIn []Message
	seenStreamIn []Message
}

func (f *fakeAgent) Invoke(ctx context.Context, in []Message) ([]Message, error) {
	f.seenInvokeIn = append([]Message(nil), in...) // snapshot, defend against later mutation
	if f.invokeErr != nil {
		return nil, f.invokeErr
	}
	return f.invokeOut, nil
}

func (f *fakeAgent) Stream(ctx context.Context, in []Message) iter.Seq2[Event[[]Message], error] {
	f.seenStreamIn = append([]Message(nil), in...)
	return func(yield func(Event[[]Message], error) bool) {
		for _, ev := range f.streamFrames {
			if !yield(ev, nil) {
				return
			}
		}
		if f.streamErr != nil {
			yield(Event[[]Message]{}, f.streamErr)
		}
	}
}

// msg is a tiny helper for building text-only Messages in tests.
func msg(role Role, text string) Message {
	return Message{Role: role, Content: []Part{Text{Text: text}}}
}

func TestStageError_ErrorAndUnwrap(t *testing.T) {
	underlying := errors.New("provider rejected request")
	se := &StageError{
		Index:   2,
		Err:     underlying,
		Partial: []Message{msg(RoleUser, "hi")},
	}

	if got := se.Error(); got != "stage 2: provider rejected request" {
		t.Errorf("Error() = %q, want %q", got, "stage 2: provider rejected request")
	}
	if !errors.Is(se, underlying) {
		t.Errorf("errors.Is should find the wrapped error")
	}
	var asTarget *StageError
	if !errors.As(se, &asTarget) || asTarget.Index != 2 {
		t.Errorf("errors.As should recover *StageError with Index=2")
	}
}

func TestSequential_PanicsOnZeroAgents(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("Sequential() should panic with zero agents")
		}
		if _, ok := r.(string); !ok {
			if _, ok := r.(error); !ok {
				t.Errorf("panic value should be string or error, got %T", r)
			}
		}
	}()
	Sequential()
}

func TestSequential_SingleAgentIsIdentity(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "ok")}}
	got := Sequential(a)
	// The returned Agent must be a itself, not a one-stage wrapper.
	if got != Agent(a) {
		t.Errorf("Sequential(a) should return a directly; got a different Agent")
	}
}

func TestSequential_InvokeThreadsTranscript(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-a")}}
	b := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-b")}}
	c := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-c")}}

	in := []Message{msg(RoleUser, "hello")}
	got, err := Sequential(a, b, c).Invoke(context.Background(), in)
	if err != nil {
		t.Fatalf("Invoke returned error: %v", err)
	}

	want := []Message{
		msg(RoleUser, "hello"),
		msg(RoleAssistant, "from-a"),
		msg(RoleAssistant, "from-b"),
		msg(RoleAssistant, "from-c"),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("transcript mismatch\n got: %v\nwant: %v", got, want)
	}

	wantA := []Message{msg(RoleUser, "hello")}
	wantB := []Message{msg(RoleUser, "hello"), msg(RoleAssistant, "from-a")}
	wantC := []Message{msg(RoleUser, "hello"), msg(RoleAssistant, "from-a"), msg(RoleAssistant, "from-b")}
	if !reflect.DeepEqual(a.seenInvokeIn, wantA) {
		t.Errorf("a.seenInvokeIn = %v, want %v", a.seenInvokeIn, wantA)
	}
	if !reflect.DeepEqual(b.seenInvokeIn, wantB) {
		t.Errorf("b.seenInvokeIn = %v, want %v", b.seenInvokeIn, wantB)
	}
	if !reflect.DeepEqual(c.seenInvokeIn, wantC) {
		t.Errorf("c.seenInvokeIn = %v, want %v", c.seenInvokeIn, wantC)
	}
}

func TestSequential_InvokeDoesNotMutateInput(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "out")}}
	in := []Message{msg(RoleUser, "hi")}
	inCopy := append([]Message(nil), in...)

	if _, err := Sequential(a, a).Invoke(context.Background(), in); err != nil {
		t.Fatalf("Invoke error: %v", err)
	}

	if !reflect.DeepEqual(in, inCopy) {
		t.Errorf("input was mutated\n got: %v\nwant: %v", in, inCopy)
	}
}

func TestSequential_InvokeWrapsStageError(t *testing.T) {
	boom := errors.New("provider blew up")
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-a")}}
	b := &fakeAgent{invokeErr: boom}
	c := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-c")}}

	in := []Message{msg(RoleUser, "hi")}
	got, err := Sequential(a, b, c).Invoke(context.Background(), in)
	if got != nil {
		t.Errorf("on error, returned transcript should be nil, got %v", got)
	}
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var se *StageError
	if !errors.As(err, &se) {
		t.Fatalf("expected *StageError, got %T: %v", err, err)
	}
	if se.Index != 1 {
		t.Errorf("StageError.Index = %d, want 1", se.Index)
	}
	if !errors.Is(se, boom) {
		t.Errorf("errors.Is should find the underlying error")
	}

	wantPartial := []Message{msg(RoleUser, "hi"), msg(RoleAssistant, "from-a")}
	if !reflect.DeepEqual(se.Partial, wantPartial) {
		t.Errorf("StageError.Partial = %v, want %v", se.Partial, wantPartial)
	}

	if c.seenInvokeIn != nil {
		t.Errorf("agent c was invoked despite earlier failure: seenInvokeIn=%v", c.seenInvokeIn)
	}
}
