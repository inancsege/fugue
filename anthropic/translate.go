package anthropic

import (
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
// Anthropic MessageParams. Returns an error for empty input.
func toAPIMessages(in []fugue.Message) ([]sdk.MessageParam, error) {
	if len(in) == 0 {
		return nil, errors.New("anthropic: at least one non-system message is required")
	}
	out := make([]sdk.MessageParam, 0, len(in))
	for _, m := range in {
		switch m.Role {
		case fugue.RoleUser:
			blocks, err := contentToBlocks(m.Content, m.ToolCalls)
			if err != nil {
				return nil, err
			}
			out = append(out, sdk.NewUserMessage(blocks...))
		case fugue.RoleAssistant:
			blocks, err := contentToBlocks(m.Content, m.ToolCalls)
			if err != nil {
				return nil, err
			}
			out = append(out, sdk.NewAssistantMessage(blocks...))
		case fugue.RoleTool:
			return nil, errors.New("anthropic: RoleTool translation not yet implemented")
		case fugue.RoleSystem:
			return nil, errors.New("anthropic: RoleSystem must be handled by splitSystem")
		default:
			return nil, fmt.Errorf("anthropic: unknown role %v", m.Role)
		}
	}
	return out, nil
}

// contentToBlocks translates fugue Parts (plus any ToolCalls) into Anthropic
// content blocks. Reasoning parts on input are dropped per spec D8.
func contentToBlocks(parts []fugue.Part, calls []fugue.ToolCall) ([]sdk.ContentBlockParamUnion, error) {
	out := make([]sdk.ContentBlockParamUnion, 0, len(parts)+len(calls))
	for _, p := range parts {
		switch v := p.(type) {
		case fugue.Text:
			out = append(out, sdk.NewTextBlock(v.Text))
		case fugue.Reasoning:
			// Spec D8: dropped silently on input.
			continue
		case fugue.Image:
			return nil, errors.New("anthropic: Image translation not yet implemented")
		default:
			return nil, fmt.Errorf("anthropic: unknown Part type %T", p)
		}
	}
	if len(calls) > 0 {
		return nil, errors.New("anthropic: tool_use translation not yet implemented")
	}
	return out, nil
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
