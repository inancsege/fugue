// Package anthropic — typed tools integration. WithTools registers fugue
// tools that the adapter dispatches inside its Invoke/Stream loop until the
// model stops emitting tool_use blocks or WithMaxSteps is exhausted.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"slices"
	"sync"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/inancsege/fugue"
)

// WithTools registers tools the model can call. Tools are dispatched
// internally by the agent until the model stops requesting them or the step
// budget (WithMaxSteps) is hit.
//
// Tool names must be unique within the agent; duplicates panic at
// construction.
func WithTools(tools ...fugue.ToolDef) Option {
	return func(c *config) {
		byName := make(map[string]fugue.ToolDef, len(tools))
		for _, t := range tools {
			if _, dup := byName[t.Name()]; dup {
				panic("anthropic.WithTools: duplicate tool name " + t.Name())
			}
			byName[t.Name()] = t
		}
		c.tools = append(c.tools, tools...)
		if c.toolByName == nil {
			c.toolByName = byName
		} else {
			for k, v := range byName {
				if _, dup := c.toolByName[k]; dup {
					panic("anthropic.WithTools: duplicate tool name " + k)
				}
				c.toolByName[k] = v
			}
		}
	}
}

// WithMaxSteps caps the number of model turns in a tool-use loop. A step is
// one Messages.New call; tool execution within a step is not counted.
// Default is 8 when tools are present, 1 otherwise. Setting MaxSteps=1
// disables the loop (one model turn, no tool dispatch even if the model
// emits tool_use).
func WithMaxSteps(n int) Option {
	return func(c *config) {
		c.maxSteps = n
	}
}

// toolResult builds a fugue.Message for a successful (or model-visible-error)
// tool dispatch. body is the JSON-encoded payload (or plain text for errors).
func toolResult(id string, body json.RawMessage, isError bool) fugue.Message {
	return fugue.Message{
		Role:       fugue.RoleTool,
		ToolCallID: id,
		Content:    []fugue.Part{fugue.Text{Text: string(body)}},
		IsError:    isError,
	}
}

// toolErrorResult is a convenience that builds an is_error tool-result with
// the given string body.
func toolErrorResult(id, msg string) fugue.Message {
	return fugue.Message{
		Role:       fugue.RoleTool,
		ToolCallID: id,
		Content:    []fugue.Part{fugue.Text{Text: msg}},
		IsError:    true,
	}
}

// invokeWithTools runs the model→tool→model loop. Called from Invoke when
// the agent has tools registered. Returns the full trace of newly-produced
// messages (assistant turns and tool results) in order.
//
// On budget exhaustion, returns the trace produced so far along with a
// *fugue.ToolLoopError. On transport-level failures (Messages.New error,
// response parse error, runTools transport error, ctx cancel), returns the
// partial trace and the underlying error.
func (a *agent) invokeWithTools(ctx context.Context, in []fugue.Message) ([]fugue.Message, error) {
	history := append([]fugue.Message(nil), in...)
	var produced []fugue.Message
	budget := a.cfg.maxSteps
	if budget <= 0 {
		budget = 8
	}

	for step := 0; step < budget; step++ {
		params, err := a.buildParams(history)
		if err != nil {
			return produced, err
		}
		resp, err := a.cfg.client.Messages.New(ctx, params)
		if err != nil {
			return produced, err
		}
		msg, err := fromAPIResponse(resp)
		if err != nil {
			return produced, err
		}
		produced = append(produced, msg)
		history = append(history, msg)

		if len(msg.ToolCalls) == 0 {
			return produced, nil
		}

		results, err := a.runTools(ctx, msg.ToolCalls)
		if err != nil {
			return produced, err
		}
		produced = append(produced, results...)
		history = append(history, results...)
	}
	return produced, &fugue.ToolLoopError{Steps: budget}
}

// runTools dispatches each ToolCall in calls. Tools run concurrently;
// results are returned in call order so the resulting tool_result blocks
// match the tool_use ordering Anthropic requires.
//
// Returns ctx.Err() if the context was canceled, or a transport-error wrap
// if a tool's transport return was non-nil. Otherwise returns the per-call
// result slice and nil error. Model-visible errors (fn errors, panics,
// unknown tools) are folded into is_error results.
func (a *agent) runTools(ctx context.Context, calls []fugue.ToolCall) ([]fugue.Message, error) {
	results := make([]fugue.Message, len(calls))
	transportErrs := make([]error, len(calls))
	var wg sync.WaitGroup
	for i, call := range calls {
		tool, ok := a.cfg.toolByName[call.Name]
		if !ok {
			results[i] = toolErrorResult(call.ID, fmt.Sprintf("unknown tool: %s", call.Name))
			continue
		}
		wg.Add(1)
		go func(i int, call fugue.ToolCall, tool fugue.ToolDef) {
			defer wg.Done()
			out, isErr, transportErr := tool.Invoke(ctx, call.Arguments)
			if transportErr != nil {
				transportErrs[i] = transportErr
				return
			}
			results[i] = toolResult(call.ID, out, isErr)
		}(i, call, tool)
	}
	wg.Wait()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	for _, e := range transportErrs {
		if e != nil {
			return nil, fmt.Errorf("anthropic: tool transport error: %w", e)
		}
	}
	return results, nil
}

// streamWithTools is the Stream-side counterpart to invokeWithTools. Each
// model turn streams its own SSE frames inline. Done=true appears only on
// the terminal frame of the terminal turn (one with no tool_use blocks).
// Delta is cumulative for the whole trace produced so far.
//
// Tool execution is silent in the stream — no synthetic frames between
// turns. Consumers detect "tools in flight" by inspecting Delta's last
// Message for ToolCalls.
func (a *agent) streamWithTools(ctx context.Context, in []fugue.Message) iter.Seq2[fugue.Event[[]fugue.Message], error] {
	return func(yield func(fugue.Event[[]fugue.Message], error) bool) {
		history := append([]fugue.Message(nil), in...)
		var produced []fugue.Message
		budget := a.cfg.maxSteps
		if budget <= 0 {
			budget = 8
		}

		for step := 0; step < budget; step++ {
			params, err := a.buildParams(history)
			if err != nil {
				yield(fugue.Event[[]fugue.Message]{}, err)
				return
			}
			stream := a.cfg.client.Messages.NewStreaming(ctx, params)
			var acc sdk.Message
			var lastMsg fugue.Message
			for stream.Next() {
				ev := stream.Current()
				if err := acc.Accumulate(ev); err != nil {
					stream.Close()
					yield(fugue.Event[[]fugue.Message]{}, err)
					return
				}
				partial, err := fromAPIResponse(&acc)
				if err != nil {
					stream.Close()
					yield(fugue.Event[[]fugue.Message]{}, err)
					return
				}
				lastMsg = partial
				isTurnEnd := ev.Type == "message_stop"
				done := isTurnEnd && len(partial.ToolCalls) == 0
				delta := append(slices.Clone(produced), partial)
				if !yield(fugue.Event[[]fugue.Message]{Delta: delta, Done: done}, nil) {
					stream.Close()
					return
				}
			}
			if err := stream.Err(); err != nil {
				stream.Close()
				yield(fugue.Event[[]fugue.Message]{}, err)
				return
			}
			stream.Close()
			produced = append(produced, lastMsg)
			history = append(history, lastMsg)

			if len(lastMsg.ToolCalls) == 0 {
				return // Done frame already sent above.
			}

			results, err := a.runTools(ctx, lastMsg.ToolCalls)
			if err != nil {
				yield(fugue.Event[[]fugue.Message]{}, err)
				return
			}
			produced = append(produced, results...)
			history = append(history, results...)
		}
		yield(fugue.Event[[]fugue.Message]{}, &fugue.ToolLoopError{Steps: budget})
	}
}
