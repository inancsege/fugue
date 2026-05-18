package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/inancsege/fugue"
)

// makeAgent builds an agent with the given tools, no transport — for
// runTools-only tests that don't hit the SDK.
func makeAgent(tools ...fugue.ToolDef) *agent {
	a := New("claude-sonnet-4-6", WithTools(tools...))
	return a.(*agent)
}

func TestRunTools_OrdersResultsByCallIndex(t *testing.T) {
	// Two tools; the slower one is called first. Result order must follow
	// call order, not completion order.
	slow := fugue.RawTool("slow", "slow tool", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			time.Sleep(30 * time.Millisecond)
			return json.RawMessage(`"slow-done"`), nil
		})
	fast := fugue.RawTool("fast", "fast tool", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`"fast-done"`), nil
		})
	a := makeAgent(slow, fast)

	results, err := a.runTools(context.Background(), []fugue.ToolCall{
		{ID: "c1", Name: "slow", Arguments: json.RawMessage(`{}`)},
		{ID: "c2", Name: "fast", Arguments: json.RawMessage(`{}`)},
	})
	if err != nil {
		t.Fatalf("runTools: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("want 2 results, got %d", len(results))
	}
	if results[0].ToolCallID != "c1" || results[1].ToolCallID != "c2" {
		t.Errorf("result order mismatched call order: %s, %s", results[0].ToolCallID, results[1].ToolCallID)
	}
}

func TestRunTools_RunsToolsInParallel(t *testing.T) {
	// Each tool waits 50ms. Sequential would take ≥100ms; parallel ~50ms.
	// We assert the wall-clock is under 90ms as a loose bound.
	var inflight atomic.Int32
	var maxInflight atomic.Int32

	mk := func(name string) fugue.ToolDef {
		return fugue.RawTool(name, name, json.RawMessage(`{"type":"object"}`),
			func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
				n := inflight.Add(1)
				for {
					m := maxInflight.Load()
					if n <= m || maxInflight.CompareAndSwap(m, n) {
						break
					}
				}
				time.Sleep(50 * time.Millisecond)
				inflight.Add(-1)
				return json.RawMessage(`"ok"`), nil
			})
	}
	a := makeAgent(mk("a"), mk("b"))

	start := time.Now()
	_, err := a.runTools(context.Background(), []fugue.ToolCall{
		{ID: "1", Name: "a", Arguments: json.RawMessage(`{}`)},
		{ID: "2", Name: "b", Arguments: json.RawMessage(`{}`)},
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("runTools: %v", err)
	}
	if elapsed > 90*time.Millisecond {
		t.Errorf("parallel dispatch should complete in <90ms, took %v", elapsed)
	}
	if maxInflight.Load() < 2 {
		t.Errorf("expected concurrent execution (maxInflight=%d, want ≥2)", maxInflight.Load())
	}
}

func TestRunTools_UnknownToolFedBackAsError(t *testing.T) {
	// Register one tool so we get a non-nil toolByName but still trigger unknown path.
	noop := fugue.RawTool("noop", "noop", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) { return nil, nil })
	a := makeAgent(noop)
	results, err := a.runTools(context.Background(), []fugue.ToolCall{
		{ID: "x", Name: "missing", Arguments: json.RawMessage(`{}`)},
	})
	if err != nil {
		t.Fatalf("runTools should not return err for unknown tool, got %v", err)
	}
	if len(results) != 1 || !results[0].IsError {
		t.Fatalf("expected one is_error result, got %+v", results)
	}
	if !strings.Contains(asText(results[0]), "missing") {
		t.Errorf("error content should mention the unknown name, got %q", asText(results[0]))
	}
}

func TestRunTools_FnPanicFedBackAsError(t *testing.T) {
	bad := fugue.RawTool("boom", "boom", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			panic("kaboom")
		})
	a := makeAgent(bad)
	results, err := a.runTools(context.Background(), []fugue.ToolCall{
		{ID: "1", Name: "boom", Arguments: json.RawMessage(`{}`)},
	})
	if err != nil {
		t.Fatalf("runTools should recover panic, got err %v", err)
	}
	if !results[0].IsError {
		t.Fatal("expected is_error after panic")
	}
	if !strings.Contains(asText(results[0]), "kaboom") {
		t.Errorf("error should contain panic message, got %q", asText(results[0]))
	}
}

func TestRunTools_FnErrorFedBackAsError(t *testing.T) {
	bad := fugue.RawTool("503", "always 503", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("upstream 503")
		})
	a := makeAgent(bad)
	results, err := a.runTools(context.Background(), []fugue.ToolCall{
		{ID: "1", Name: "503", Arguments: json.RawMessage(`{}`)},
	})
	if err != nil {
		t.Fatalf("runTools: %v", err)
	}
	if !results[0].IsError || !strings.Contains(asText(results[0]), "upstream 503") {
		t.Errorf("expected is_error with upstream 503 message, got %+v", results[0])
	}
}

func TestRunTools_CtxCancelReturnsErr(t *testing.T) {
	hung := fugue.RawTool("hung", "hangs", json.RawMessage(`{"type":"object"}`),
		func(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		})
	a := makeAgent(hung)
	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()
	_, err := a.runTools(ctx, []fugue.ToolCall{
		{ID: "1", Name: "hung", Arguments: json.RawMessage(`{}`)},
	})
	wg.Wait()
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

// asText pulls the first Text part from a tool-result message.
func asText(m fugue.Message) string {
	for _, p := range m.Content {
		if t, ok := p.(fugue.Text); ok {
			return t.Text
		}
	}
	return ""
}
