package fugue

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestRawTool_HappyPath(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`)
	tool := RawTool("search", "Search the corpus", schema, func(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
		var in struct{ Q string `json:"q"` }
		if err := json.Unmarshal(args, &in); err != nil {
			return nil, err
		}
		return json.RawMessage(`{"hits":["` + in.Q + `"]}`), nil
	})

	if tool.Name() != "search" {
		t.Errorf("Name = %q, want search", tool.Name())
	}
	if tool.Description() != "Search the corpus" {
		t.Errorf("Description = %q, want Search the corpus", tool.Description())
	}
	if string(tool.Schema()) != string(schema) {
		t.Errorf("Schema mismatch")
	}

	out, isErr, transportErr := tool.Invoke(context.Background(), json.RawMessage(`{"q":"hello"}`))
	if transportErr != nil {
		t.Fatalf("transportErr: %v", transportErr)
	}
	if isErr {
		t.Errorf("isErr = true, want false")
	}
	if string(out) != `{"hits":["hello"]}` {
		t.Errorf("out = %s, want hits[hello]", out)
	}
}

func TestRawTool_FnErrorBecomesIsError(t *testing.T) {
	tool := RawTool("boom", "always fails", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("rate limited")
		})

	out, isErr, transportErr := tool.Invoke(context.Background(), json.RawMessage(`{}`))
	if transportErr != nil {
		t.Fatalf("transportErr: %v", transportErr)
	}
	if !isErr {
		t.Fatal("isErr = false, want true")
	}
	if string(out) != "rate limited" {
		t.Errorf("out = %q, want %q", out, "rate limited")
	}
}

func TestRawTool_PanicsOnEmptyName(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		if !strings.Contains(r.(string), "name") {
			t.Errorf("panic should mention name, got: %v", r)
		}
	}()
	RawTool("", "desc", json.RawMessage(`{}`), func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) { return nil, nil })
}

func TestRawTool_PanicsOnEmptyDescription(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	RawTool("name", "", json.RawMessage(`{}`), func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) { return nil, nil })
}

func TestRawTool_PanicsOnNilFn(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic")
		}
	}()
	RawTool("name", "desc", json.RawMessage(`{}`), nil)
}

func TestToolDef_InvokeRecoversPanic(t *testing.T) {
	tool := RawTool("boom", "panics", json.RawMessage(`{}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			panic("kaboom")
		})

	out, isErr, transportErr := tool.Invoke(context.Background(), json.RawMessage(`{}`))
	if transportErr != nil {
		t.Fatalf("transportErr should be nil after recovery, got %v", transportErr)
	}
	if !isErr {
		t.Fatal("isErr = false, want true after panic")
	}
	if !strings.Contains(string(out), "kaboom") {
		t.Errorf("recovered panic message should contain 'kaboom', got %q", out)
	}
}
