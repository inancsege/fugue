package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/inancsege/fugue"
)

// splitSystem extracts the system prompt from the input messages and returns
// the body messages stripped of any leading RoleSystem entries.
//
// Resolution: optionPrompt wins if non-empty; otherwise consecutive leading
// RoleSystem messages are concatenated with newlines. A RoleSystem message
// that appears after any non-system message returns an error — Anthropic's
// API does not support mid-conversation system messages.
func splitSystem(in []fugue.Message, optionPrompt string) (system string, body []fugue.Message, err error) {
	body = in
	var leading []string
	for i, m := range in {
		if m.Role == fugue.RoleSystem {
			leading = append(leading, partsToText(m.Content))
			continue
		}
		body = in[i:]
		break
	}
	if len(leading) == len(in) {
		body = nil
	}

	for _, m := range body {
		if m.Role == fugue.RoleSystem {
			return "", nil, errors.New("anthropic: RoleSystem must lead the conversation; the API only supports a top-level system parameter")
		}
	}

	if optionPrompt != "" {
		return optionPrompt, body, nil
	}
	return strings.Join(leading, "\n"), body, nil
}

// partsToText extracts only Text parts from a message's content, joined with
// no separator. Used by splitSystem since system prompts are plain strings.
func partsToText(parts []fugue.Part) string {
	var b strings.Builder
	for _, p := range parts {
		if t, ok := p.(fugue.Text); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

// toAPIMessages translates a fugue conversation body (post-splitSystem) into
// Anthropic MessageParams. Returns an error for empty input. Consecutive
// RoleTool messages collapse into a single user-role message containing
// multiple tool_result blocks (Anthropic's required shape).
func toAPIMessages(in []fugue.Message) ([]sdk.MessageParam, error) {
	if len(in) == 0 {
		return nil, errors.New("anthropic: at least one non-system message is required")
	}
	out := make([]sdk.MessageParam, 0, len(in))
	i := 0
	for i < len(in) {
		m := in[i]
		switch m.Role {
		case fugue.RoleUser:
			blocks, err := contentToBlocks(m.Content, m.ToolCalls)
			if err != nil {
				return nil, err
			}
			out = append(out, sdk.NewUserMessage(blocks...))
			i++
		case fugue.RoleAssistant:
			blocks, err := contentToBlocks(m.Content, m.ToolCalls)
			if err != nil {
				return nil, err
			}
			out = append(out, sdk.NewAssistantMessage(blocks...))
			i++
		case fugue.RoleTool:
			var blocks []sdk.ContentBlockParamUnion
			for i < len(in) && in[i].Role == fugue.RoleTool {
				tm := in[i]
				if tm.ToolCallID == "" {
					return nil, errors.New("anthropic: RoleTool message requires ToolCallID")
				}
				blocks = append(blocks, sdk.NewToolResultBlock(tm.ToolCallID, partsToText(tm.Content), tm.IsError))
				i++
			}
			out = append(out, sdk.NewUserMessage(blocks...))
		case fugue.RoleSystem:
			return nil, errors.New("anthropic: RoleSystem must be handled by splitSystem")
		default:
			return nil, fmt.Errorf("anthropic: unknown role %v", m.Role)
		}
	}
	return out, nil
}

// contentToBlocks translates fugue Parts (plus any ToolCalls) into Anthropic
// content blocks. Reasoning parts on input are dropped silently — Anthropic's
// extended-thinking blocks are signed by the API and cannot be reconstructed
// faithfully across the framework boundary, so re-sending them as text would
// either fail signature verification or pollute the conversation. The model
// will produce fresh thinking blocks on the next turn if thinking is enabled.
func contentToBlocks(parts []fugue.Part, calls []fugue.ToolCall) ([]sdk.ContentBlockParamUnion, error) {
	out := make([]sdk.ContentBlockParamUnion, 0, len(parts)+len(calls))
	for _, p := range parts {
		switch v := p.(type) {
		case fugue.Text:
			out = append(out, sdk.NewTextBlock(v.Text))
		case fugue.Reasoning:
			continue
		case fugue.Image:
			block, err := imageBlock(v)
			if err != nil {
				return nil, err
			}
			out = append(out, block)
		default:
			return nil, fmt.Errorf("anthropic: unknown Part type %T", p)
		}
	}
	for _, tc := range calls {
		// SDK's NewToolUseBlock wraps Input as `any` — pass json.RawMessage
		// through; it implements MarshalJSON, so the SDK encodes it verbatim.
		out = append(out, sdk.NewToolUseBlock(tc.ID, tc.Arguments, tc.Name))
	}
	return out, nil
}

// imageBlock translates a fugue.Image into an Anthropic image content block.
// Per spec: Data wins over URL when both are set.
func imageBlock(img fugue.Image) (sdk.ContentBlockParamUnion, error) {
	switch {
	case len(img.Data) > 0:
		if img.MIMEType == "" {
			return sdk.ContentBlockParamUnion{}, errors.New("anthropic: Image with Data requires non-empty MIMEType")
		}
		return sdk.NewImageBlockBase64(img.MIMEType, base64.StdEncoding.EncodeToString(img.Data)), nil
	case img.URL != "":
		return sdk.NewImageBlock(sdk.URLImageSourceParam{URL: img.URL}), nil
	default:
		return sdk.ContentBlockParamUnion{}, errors.New("anthropic: Image requires Data or URL")
	}
}

// fromAPIResponse translates an Anthropic Messages API response into a single
// assistant fugue.Message. Text blocks become Text parts, thinking blocks
// become Reasoning parts, tool_use blocks become ToolCalls. The stop_reason
// is exposed via Message.Name.
func fromAPIResponse(resp *sdk.Message) (fugue.Message, error) {
	out := fugue.Message{Role: fugue.RoleAssistant}
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			out.Content = append(out.Content, fugue.Text{Text: block.Text})
		case "thinking":
			out.Content = append(out.Content, fugue.Reasoning{Text: block.Thinking})
		case "tool_use":
			out.ToolCalls = append(out.ToolCalls, fugue.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: block.Input,
			})
		case "redacted_thinking":
			// Drop silently — caller cannot meaningfully use redacted content.
			continue
		default:
			return fugue.Message{}, fmt.Errorf("anthropic: unsupported response block type %q", block.Type)
		}
	}
	out.Name = string(resp.StopReason)
	return out, nil
}

// toolDefToSDKToolUnion translates a fugue.ToolDef into an Anthropic
// ToolUnionParam (variant: OfTool). The ToolDef's schema is the full
// JSON-Schema object {type:object, properties:..., required:...}; we parse
// it once and place properties/required into the SDK's typed shape.
//
// An invalid schema (cannot parse as a JSON object, or properties not an
// object) returns an error. Callers (the anthropic agent constructor) should
// panic on this — it indicates a buggy RawTool schema, not a runtime
// condition.
func toolDefToSDKToolUnion(td fugue.ToolDef) (sdk.ToolUnionParam, error) {
	schema := td.Schema()
	if len(schema) == 0 {
		// No schema (e.g., RawTool called with nil schema) — emit empty
		// object schema so the API accepts it.
		schema = json.RawMessage(`{"type":"object"}`)
	}
	var parsed struct {
		Properties json.RawMessage `json:"properties"`
		Required   []string        `json:"required"`
	}
	if err := json.Unmarshal(schema, &parsed); err != nil {
		return sdk.ToolUnionParam{}, fmt.Errorf("anthropic: tool %q has invalid JSON schema: %w", td.Name(), err)
	}
	var props any
	if len(parsed.Properties) > 0 {
		var m map[string]json.RawMessage
		if err := json.Unmarshal(parsed.Properties, &m); err != nil {
			return sdk.ToolUnionParam{}, fmt.Errorf("anthropic: tool %q properties must be a JSON object: %w", td.Name(), err)
		}
		props = m
	}
	param := sdk.ToolParam{
		Name: td.Name(),
		InputSchema: sdk.ToolInputSchemaParam{
			Properties: props,
			Required:   parsed.Required,
		},
	}
	if td.Description() != "" {
		param.Description = sdk.String(td.Description())
	}
	return sdk.ToolUnionParam{OfTool: &param}, nil
}
