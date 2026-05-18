// Package anthropic — typed tools integration. WithTools registers fugue
// tools that the adapter dispatches inside its Invoke/Stream loop until the
// model stops emitting tool_use blocks or WithMaxSteps is exhausted.
package anthropic

import (
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
