package fugue

import (
	"context"
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
