// Package anthropic — typed tools integration. WithTools registers fugue
// tools that the adapter dispatches inside its Invoke/Stream loop until the
// model stops emitting tool_use blocks or WithMaxSteps is exhausted.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

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
