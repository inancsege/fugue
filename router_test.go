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
