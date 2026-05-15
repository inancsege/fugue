package fugue

import (
	"errors"
	"testing"
)

func TestToolLoopError_FormatAndAs(t *testing.T) {
	var loop *ToolLoopError
	err := error(&ToolLoopError{Steps: 8})
	if !errors.As(err, &loop) {
		t.Fatal("errors.As(*ToolLoopError) should match")
	}
	if loop.Steps != 8 {
		t.Errorf("Steps = %d, want 8", loop.Steps)
	}
	want := "fugue: tool-use loop exceeded 8 steps"
	if got := err.Error(); got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}
