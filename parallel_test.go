package fugue

import (
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
