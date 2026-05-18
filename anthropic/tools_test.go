package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
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

// toolUseResponse builds a 200 JSON response with one tool_use block.
func toolUseResponse(toolName, toolID, args string) *http.Response {
	body := `{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-4-6",
	"content":[{"type":"tool_use","id":"` + toolID + `","name":"` + toolName + `","input":` + args + `}],
	"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
	return okResponse(body)
}

// textResponse builds a 200 JSON response with a final text block.
func textResponse(text string) *http.Response {
	body := `{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-4-6",
	"content":[{"type":"text","text":"` + text + `"}],
	"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	return okResponse(body)
}

func TestInvokeWithTools_NoToolUseReturnsImmediately(t *testing.T) {
	tool := fugue.RawTool("noop", "noop", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) { return nil, nil })
	ft := &fakeTransport{responses: []*http.Response{textResponse("hello")}}
	a := newAgentWithTransport("claude-sonnet-4-6", ft, WithTools(tool))

	out, err := a.Invoke(context.Background(), []fugue.Message{msg(fugue.RoleUser, "hi")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("want 1 message, got %d", len(out))
	}
	if len(ft.requests) != 1 {
		t.Errorf("want 1 HTTP request, got %d", len(ft.requests))
	}
}

func TestInvokeWithTools_OneToolThenFinalText(t *testing.T) {
	tool := fugue.RawTool("search", "search", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"hits":["match"]}`), nil
		})
	ft := &fakeTransport{responses: []*http.Response{
		toolUseResponse("search", "toolu_1", `{"query":"x"}`),
		textResponse("found one match"),
	}}
	a := newAgentWithTransport("claude-sonnet-4-6", ft, WithTools(tool))

	out, err := a.Invoke(context.Background(), []fugue.Message{msg(fugue.RoleUser, "find x")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	// Trace: assistant(tool_use), tool(result), assistant(final).
	if len(out) != 3 {
		t.Fatalf("want 3 messages in trace, got %d: %+v", len(out), out)
	}
	if out[0].Role != fugue.RoleAssistant || len(out[0].ToolCalls) != 1 {
		t.Errorf("out[0] should be assistant with one tool_use, got %+v", out[0])
	}
	if out[1].Role != fugue.RoleTool || out[1].ToolCallID != "toolu_1" {
		t.Errorf("out[1] should be tool result, got %+v", out[1])
	}
	if out[2].Role != fugue.RoleAssistant || asText(out[2]) != "found one match" {
		t.Errorf("out[2] should be final assistant text, got %+v", out[2])
	}
	if len(ft.requests) != 2 {
		t.Errorf("want 2 HTTP requests, got %d", len(ft.requests))
	}
}

func TestInvokeWithTools_ToolLoopErrorOnBudgetExhaustion(t *testing.T) {
	tool := fugue.RawTool("loop", "loop", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{}`), nil
		})
	// Always return tool_use → never converges.
	resps := []*http.Response{
		toolUseResponse("loop", "u1", `{}`),
		toolUseResponse("loop", "u2", `{}`),
		toolUseResponse("loop", "u3", `{}`),
	}
	ft := &fakeTransport{responses: resps}
	a := newAgentWithTransport("claude-sonnet-4-6", ft, WithTools(tool), WithMaxSteps(3))

	_, err := a.Invoke(context.Background(), []fugue.Message{msg(fugue.RoleUser, "go")})
	var loopErr *fugue.ToolLoopError
	if !errors.As(err, &loopErr) {
		t.Fatalf("want *ToolLoopError, got %T: %v", err, err)
	}
	if loopErr.Steps != 3 {
		t.Errorf("Steps = %d, want 3", loopErr.Steps)
	}
}

// toolUseSSEResponse builds a streaming SSE response that emits one tool_use
// block (input arrives in two json_delta chunks) and ends with message_stop.
func toolUseSSEResponse(toolName, toolID string) *http.Response {
	events := []string{
		`event: message_start
data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}

`,
		`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"` + toolID + `","name":"` + toolName + `","input":{}}}

`,
		`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\""}}

`,
		`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"x\"}"}}

`,
		`event: content_block_stop
data: {"type":"content_block_stop","index":0}

`,
		`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":2}}

`,
		`event: message_stop
data: {"type":"message_stop"}

`,
	}
	return sseResponse(events...)
}

func TestStreamWithTools_DoneOnlyOnFinalTurn(t *testing.T) {
	tool := fugue.RawTool("search", "search", json.RawMessage(`{"type":"object"}`),
		func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"hits":[]}`), nil
		})
	ft := &fakeTransport{responses: []*http.Response{
		toolUseSSEResponse("search", "u1"),
		streamingResponseTwoTextDeltas(),
	}}
	a := newAgentWithTransport("claude-sonnet-4-6", ft, WithTools(tool))

	var frames []fugue.Event[[]fugue.Message]
	for ev, err := range a.Stream(context.Background(), []fugue.Message{msg(fugue.RoleUser, "go")}) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		frames = append(frames, ev)
	}
	if len(frames) == 0 {
		t.Fatal("expected frames")
	}
	for i, f := range frames[:len(frames)-1] {
		if f.Done {
			t.Errorf("frame %d should have Done=false (intermediate turn)", i)
		}
	}
	if !frames[len(frames)-1].Done {
		t.Errorf("final frame must have Done=true")
	}
	// Final cumulative trace must contain the tool_use turn, the tool_result,
	// and the final assistant text.
	final := frames[len(frames)-1].Delta
	if len(final) < 3 {
		t.Fatalf("final trace too short, got %d messages: %+v", len(final), final)
	}
	if final[0].Role != fugue.RoleAssistant || len(final[0].ToolCalls) == 0 {
		t.Errorf("final[0] should be assistant tool_use, got %+v", final[0])
	}
	if final[1].Role != fugue.RoleTool {
		t.Errorf("final[1] should be tool result, got %+v", final[1])
	}
}
