package fugue

import "encoding/json"

// Role identifies the speaker of a [Message].
type Role uint8

const (
	RoleUser Role = iota + 1
	RoleAssistant
	RoleSystem
	RoleTool
)

func (r Role) String() string {
	switch r {
	case RoleUser:
		return "user"
	case RoleAssistant:
		return "assistant"
	case RoleSystem:
		return "system"
	case RoleTool:
		return "tool"
	default:
		return "unknown"
	}
}

// Message is the unit of conversation between agents and providers.
//
// fugue uses a single struct with role discrimination — matching the dominant
// framework-layer pattern (Eino schema.Message, Genkit Go ai.Message). Multimodal
// content is modeled as a slice of [Part], a sealed interface, so provider adapters
// and renderers can switch exhaustively on content kind.
type Message struct {
	Role       Role
	Content    []Part
	ToolCalls  []ToolCall
	ToolCallID string // set when Role == RoleTool
	Name       string
}

// Part is a sealed interface for message content blocks.
//
// New content kinds are added by implementing this interface. Provider adapters
// should switch exhaustively on the concrete type.
type Part interface {
	isPart()
}

// Text is a plain-text content block.
type Text struct {
	Text string
}

// Image is an image content block. Either Data (with MIMEType) or URL must be set.
type Image struct {
	MIMEType string
	Data     []byte
	URL      string
}

// Reasoning is a model reasoning / chain-of-thought block.
//
// Surfaced separately from [Text] so adapters can choose to forward, hide, or
// persist it independently of the user-visible response.
type Reasoning struct {
	Text string
}

func (Text) isPart()      {}
func (Image) isPart()     {}
func (Reasoning) isPart() {}

// ToolCall is a request from an assistant to invoke a named tool with JSON arguments.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}
