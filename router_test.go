package fugue

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestRouter_PanicsOnNilDecide(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Router(nil, ...) should panic")
		}
	}()
	Router(nil, map[string]Agent{"a": &fakeAgent{}})
}

func TestRouter_PanicsOnNilRoutes(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Router(decide, nil) should panic")
		}
	}()
	decide := func(ctx context.Context, in []Message) (string, error) { return "x", nil }
	Router(decide, nil)
}

func TestRouter_PanicsOnEmptyRoutes(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Router(decide, emptyMap) should panic")
		}
	}()
	decide := func(ctx context.Context, in []Message) (string, error) { return "x", nil }
	Router(decide, map[string]Agent{})
}

func TestRouteError_ErrorAndUnwrap(t *testing.T) {
	underlying := errors.New("classifier offline")

	// With Key set: message includes the key.
	re := &RouteError{Key: "plan", Err: underlying}
	if !strings.Contains(re.Error(), "plan") || !strings.Contains(re.Error(), "classifier offline") {
		t.Errorf("Error() = %q, should contain key and underlying", re.Error())
	}
	if !errors.Is(re, underlying) {
		t.Errorf("errors.Is should find the wrapped error")
	}

	// With Key empty (decide-error case): message still surfaces underlying.
	re2 := &RouteError{Key: "", Err: underlying}
	if !strings.Contains(re2.Error(), "classifier offline") {
		t.Errorf("Error() should surface underlying even with empty Key, got %q", re2.Error())
	}
	if !errors.Is(re2, underlying) {
		t.Errorf("errors.Is should still find the wrapped error with empty Key")
	}
}

func TestRouter_InvokeDispatchesToChosenAgent(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-a")}}
	b := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-b")}}

	decide := func(ctx context.Context, in []Message) (string, error) {
		return "a", nil
	}
	r := Router(decide, map[string]Agent{"a": a, "b": b})

	in := []Message{msg(RoleUser, "hi")}
	got, err := r.Invoke(context.Background(), in)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	want := []Message{msg(RoleAssistant, "from-a")}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("output mismatch\n got: %v\nwant: %v", got, want)
	}
	if b.seenInvokeIn != nil {
		t.Errorf("agent b should not have been invoked, saw %v", b.seenInvokeIn)
	}
}

func TestRouter_InvokeDispatchesPerCall(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-a")}}
	b := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-b")}}

	calls := 0
	decide := func(ctx context.Context, in []Message) (string, error) {
		calls++
		if calls%2 == 1 {
			return "a", nil
		}
		return "b", nil
	}
	r := Router(decide, map[string]Agent{"a": a, "b": b})

	in := []Message{msg(RoleUser, "hi")}
	out1, err := r.Invoke(context.Background(), in)
	if err != nil {
		t.Fatalf("Invoke #1: %v", err)
	}
	out2, err := r.Invoke(context.Background(), in)
	if err != nil {
		t.Fatalf("Invoke #2: %v", err)
	}

	if !reflect.DeepEqual(out1, []Message{msg(RoleAssistant, "from-a")}) {
		t.Errorf("out1 = %v, want from-a", out1)
	}
	if !reflect.DeepEqual(out2, []Message{msg(RoleAssistant, "from-b")}) {
		t.Errorf("out2 = %v, want from-b", out2)
	}
}

func TestRouter_InvokeWrapsDecideError(t *testing.T) {
	boom := errors.New("classifier offline")
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-a")}}

	decide := func(ctx context.Context, in []Message) (string, error) {
		return "", boom
	}
	r := Router(decide, map[string]Agent{"a": a})

	_, err := r.Invoke(context.Background(), []Message{msg(RoleUser, "hi")})
	if err == nil {
		t.Fatal("expected error")
	}

	var re *RouteError
	if !errors.As(err, &re) {
		t.Fatalf("expected *RouteError, got %T: %v", err, err)
	}
	if re.Key != "" {
		t.Errorf("RouteError.Key = %q, want empty (decide-error case)", re.Key)
	}
	if !errors.Is(re, boom) {
		t.Error("errors.Is should find the underlying error")
	}
	if a.seenInvokeIn != nil {
		t.Errorf("agent should not have been invoked when decide errored")
	}
}

func TestRouter_InvokeWrapsUnknownKey(t *testing.T) {
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "from-a")}}

	decide := func(ctx context.Context, in []Message) (string, error) {
		return "missing", nil
	}
	r := Router(decide, map[string]Agent{"a": a})

	_, err := r.Invoke(context.Background(), []Message{msg(RoleUser, "hi")})
	if err == nil {
		t.Fatal("expected error")
	}

	var re *RouteError
	if !errors.As(err, &re) {
		t.Fatalf("expected *RouteError, got %T: %v", err, err)
	}
	if re.Key != "missing" {
		t.Errorf("RouteError.Key = %q, want %q", re.Key, "missing")
	}
	if a.seenInvokeIn != nil {
		t.Errorf("agent a should not have been invoked when key was unknown")
	}
}

func TestRouter_InvokeAgentErrorIsBare(t *testing.T) {
	boom := errors.New("agent blew up")
	failing := &fakeAgent{invokeErr: boom}

	decide := func(ctx context.Context, in []Message) (string, error) {
		return "fail", nil
	}
	r := Router(decide, map[string]Agent{"fail": failing})

	_, err := r.Invoke(context.Background(), []Message{msg(RoleUser, "hi")})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, boom) {
		t.Errorf("errors.Is should find boom, got: %v", err)
	}

	var re *RouteError
	if errors.As(err, &re) {
		t.Errorf("agent error should not be wrapped in *RouteError, got: %+v", re)
	}
}

func TestRouter_DecideReceivesCallerContextAndInput(t *testing.T) {
	type ctxKey struct{}
	ctx := context.WithValue(context.Background(), ctxKey{}, "marker")
	in := []Message{msg(RoleUser, "hi")}

	var seenCtx context.Context
	var seenIn []Message
	decide := func(ctx context.Context, msgs []Message) (string, error) {
		seenCtx = ctx
		seenIn = msgs
		return "a", nil
	}
	a := &fakeAgent{invokeOut: []Message{msg(RoleAssistant, "ok")}}
	r := Router(decide, map[string]Agent{"a": a})

	if _, err := r.Invoke(ctx, in); err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	if seenCtx == nil || seenCtx.Value(ctxKey{}) != "marker" {
		t.Errorf("decide did not receive the caller's context")
	}
	if !reflect.DeepEqual(seenIn, in) {
		t.Errorf("decide saw %v, want %v", seenIn, in)
	}
	if !reflect.DeepEqual(a.seenInvokeIn, in) {
		t.Errorf("chosen agent saw %v, want %v", a.seenInvokeIn, in)
	}
}
