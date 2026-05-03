package anthropic

import (
	"errors"
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
	return nil, errors.New("anthropic: toAPIMessages not yet implemented for non-empty input")
}
