package fugue

import (
	"context"
	"reflect"
	"testing"
)

func TestParallel_PanicsOnZeroAgents(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Parallel() should panic with zero agents")
		}
	}()
	Parallel()
}

func TestParallel_SingleAgentIsIdentity(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "ok")}}
	got := Parallel(a)
	if got != Agent(a) {
		t.Errorf("Parallel(a) should return a directly; got a different Agent")
	}
}

func TestParallel_InvokeThreadsOutputsInIndexOrder(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-a")}}
	b := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-b")}}
	c := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-c")}}

	in := []Message{msg(RoleUser, "hello")}
	got, err := Parallel(a, b, c).Invoke(context.Background(), in)
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
}
