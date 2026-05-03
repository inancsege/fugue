package fugue

import (
	"context"
	"errors"
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
