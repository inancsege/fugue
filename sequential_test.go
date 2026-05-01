package fugue

import (
	"context"
	"errors"
	"iter"
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
