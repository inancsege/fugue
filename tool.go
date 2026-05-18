package fugue

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
)

// ToolDef is a tool definition bound to an agent at construction.
//
// Construct via [Tool] (typed function, schema reflected from In) or
// [RawTool] (raw JSON-in/JSON-out, schema hand-written). Providers translate
// ToolDef into their wire format via their adapter.
type ToolDef struct {
	name        string
	description string
	schema      json.RawMessage
	invoke      func(context.Context, json.RawMessage) (out json.RawMessage, isError bool, transportErr error)
}

// Name returns the tool's name. Visible to the model.
func (t ToolDef) Name() string { return t.name }

// Description returns the tool's description. Visible to the model.
func (t ToolDef) Description() string { return t.description }

// Schema returns the tool's JSON Schema as raw JSON. Provider adapters
// translate this into their wire-format schema.
func (t ToolDef) Schema() json.RawMessage { return t.schema }

// Invoke runs the tool with the given JSON-encoded arguments.
//
// Return semantics:
//   - (out, false, nil)   — success; out is the JSON-encoded result that the
//     model will see as the tool_result body.
//   - (msg, true, nil)    — model-visible failure; msg becomes the
//     tool_result body with is_error=true. The model can retry, fall back,
//     or apologize. Used for: tool fn returned err, args failed to unmarshal,
//     tool panicked.
//   - (nil, _, err)       — transport error; bubbles out of the provider's
//     Invoke. Used for unrecoverable framework-internal failures (e.g.
//     json.Marshal of a non-marshalable Out type — a programming bug).
//
// Invoke must never panic; the typed constructors install recover guards.
func (t ToolDef) Invoke(ctx context.Context, args json.RawMessage) (json.RawMessage, bool, error) {
	return t.invoke(ctx, args)
}

// Tool wraps a typed Go function as a fugue tool.
//
// The In type is reflected into a JSON Schema (see reflectInputSchema for
// supported types and tag conventions). Out is JSON-marshaled when the tool
// returns. Both In and Out should be struct types — pass a json.RawMessage
// field or use RawTool for non-struct shapes.
//
// Semantics:
//   - args fail to unmarshal into In → tool_result{is_error:true, content: err.Error()}
//   - fn returns non-nil error      → tool_result{is_error:true, content: err.Error()}
//   - fn panics                      → tool_result{is_error:true, content: "tool panic: ..."}
//   - out fails to marshal           → transport error (bubbles out of Invoke;
//                                       this is a programming bug — Out contains
//                                       a chan, func, etc.)
//
// Tool panics at construction if In's schema cannot be reflected, if name or
// description is empty, or if fn is nil.
func Tool[In, Out any](
	name, description string,
	fn func(ctx context.Context, in In) (Out, error),
) ToolDef {
	if name == "" {
		panic("fugue.Tool: name must not be empty")
	}
	if description == "" {
		panic("fugue.Tool: description must not be empty")
	}
	if fn == nil {
		panic("fugue.Tool: fn must not be nil")
	}
	var zero In
	schema, err := reflectInputSchema(reflect.TypeOf(zero))
	if err != nil {
		// reflectInputSchema errors already include the "fugue.Tool:" prefix.
		panic(err.Error())
	}

	return ToolDef{
		name:        name,
		description: description,
		schema:      schema,
		invoke: func(ctx context.Context, args json.RawMessage) (out json.RawMessage, isError bool, transportErr error) {
			defer func() {
				if r := recover(); r != nil {
					out = json.RawMessage(fmt.Sprintf("tool panic: %v", r))
					isError = true
					transportErr = nil
				}
			}()
			var in In
			if err := json.Unmarshal(args, &in); err != nil {
				return json.RawMessage(err.Error()), true, nil
			}
			result, err := fn(ctx, in)
			if err != nil {
				return json.RawMessage(err.Error()), true, nil
			}
			outBytes, err := json.Marshal(result)
			if err != nil {
				return nil, false, fmt.Errorf("fugue.Tool %q: marshal output: %w", name, err)
			}
			return outBytes, false, nil
		},
	}
}

// RawTool wraps a raw JSON-in/JSON-out function as a fugue tool.
//
// Use when [Tool]'s reflection cannot express the schema you need (recursive
// types, provider-specific JSON Schema features) or when the tool's output is
// not naturally a Go struct. The caller is responsible for the schema; this
// constructor does not validate it.
//
// fn errors become tool_result{is_error: true, content: err.Error()}; the
// model sees the failure and may retry.
//
// RawTool panics at construction if name, description, or fn is empty/nil —
// these are programming bugs.
func RawTool(
	name, description string,
	schema json.RawMessage,
	fn func(ctx context.Context, args json.RawMessage) (json.RawMessage, error),
) ToolDef {
	if name == "" {
		panic("fugue.RawTool: name must not be empty")
	}
	if description == "" {
		panic("fugue.RawTool: description must not be empty")
	}
	if fn == nil {
		panic("fugue.RawTool: fn must not be nil")
	}
	return ToolDef{
		name:        name,
		description: description,
		schema:      schema,
		invoke: func(ctx context.Context, args json.RawMessage) (out json.RawMessage, isError bool, transportErr error) {
			defer func() {
				if r := recover(); r != nil {
					out = json.RawMessage(fmt.Sprintf("tool panic: %v", r))
					isError = true
					transportErr = nil
				}
			}()
			result, err := fn(ctx, args)
			if err != nil {
				return json.RawMessage(err.Error()), true, nil
			}
			return result, false, nil
		},
	}
}
