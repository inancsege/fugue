package fugue

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
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

func TestParallel_EachAgentSeesOriginalInput(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-a")}}
	b := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-b")}}
	c := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-c")}}

	in := []Message{msg(RoleUser, "hello")}
	if _, err := Parallel(a, b, c).Invoke(context.Background(), in); err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	for name, fa := range map[string]*fakeAgent{"a": a, "b": b, "c": c} {
		if !reflect.DeepEqual(fa.seenInvokeIn, in) {
			t.Errorf("agent %s saw %v, want original input %v", name, fa.seenInvokeIn, in)
		}
	}
}

func TestParallel_InvokeWrapsStageError(t *testing.T) {
	boom := errors.New("provider blew up")
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-a")}}
	b := &fakeAgent{invokeErr: boom}
	c := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-c")}}

	in := []Message{msg(RoleUser, "hi")}
	got, err := Parallel(a, b, c).Invoke(context.Background(), in)
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
	if !reflect.DeepEqual(se.Partial, in) {
		t.Errorf("StageError.Partial = %v, want input %v", se.Partial, in)
	}
}

func TestParallel_ErrorCancelsSiblingContexts(t *testing.T) {
	boom := errors.New("explode")
	var siblingSawCancel bool
	slow := agentFunc(func(ctx context.Context, in []Message) ([]Message, error) {
		select {
		case <-time.After(2 * time.Second):
			return []Message{msg(RoleAssistant, "shouldn't reach")}, nil
		case <-ctx.Done():
			siblingSawCancel = true
			return nil, ctx.Err()
		}
	})
	fast := agentFunc(func(ctx context.Context, in []Message) ([]Message, error) {
		return nil, boom
	})

	_, err := Parallel(slow, fast).Invoke(context.Background(), []Message{msg(RoleUser, "x")})
	if err == nil {
		t.Fatal("expected error from fast stage")
	}
	if !siblingSawCancel {
		t.Error("slow sibling should have observed ctx.Done() after fast failed")
	}
}

func TestParallel_InvokePropagatesCallerContextCancel(t *testing.T) {
	a := agentFunc(func(ctx context.Context, in []Message) ([]Message, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(2 * time.Second):
			return []Message{msg(RoleAssistant, "x")}, nil
		}
	})
	b := agentFunc(func(ctx context.Context, in []Message) ([]Message, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(2 * time.Second):
			return []Message{msg(RoleAssistant, "y")}, nil
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	_, err := Parallel(a, b).Invoke(ctx, []Message{msg(RoleUser, "x")})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	var se *StageError
	if !errors.As(err, &se) {
		t.Fatalf("expected *StageError, got %T: %v", err, err)
	}
	if !errors.Is(se, context.Canceled) {
		t.Errorf("errors.Is should find context.Canceled, got: %v", se.Err)
	}
}

func TestParallel_OutputOrderIsDeterministicDespiteTiming(t *testing.T) {
	slow := agentFunc(func(ctx context.Context, in []Message) ([]Message, error) {
		select {
		case <-time.After(50 * time.Millisecond):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return []Message{msg(RoleAssistant, "slow")}, nil
	})
	fast := agentFunc(func(ctx context.Context, in []Message) ([]Message, error) {
		return []Message{msg(RoleAssistant, "fast")}, nil
	})

	in := []Message{msg(RoleUser, "go")}
	got, err := Parallel(slow, fast).Invoke(context.Background(), in)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	want := []Message{
		msg(RoleUser, "go"),
		msg(RoleAssistant, "slow"),
		msg(RoleAssistant, "fast"),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ordering mismatch\n got: %v\nwant: %v", got, want)
	}
}
