package anthropic

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/inancsege/fugue"
)

// fakeTransport returns canned responses in order. Captures requests for
// assertion. Each test constructs one inline with the responses it needs.
type fakeTransport struct {
	responses     []*http.Response
	requests      []*http.Request
	requestBodies [][]byte // captured before the body is consumed
	cursor        int

	// closeCalls tracks how many response bodies have been Close()d. Used to
	// verify Stream releases the connection on early termination.
	closeCalls int
}

func (t *fakeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, fmt.Errorf("fakeTransport: read body: %w", err)
		}
		req.Body.Close()
		t.requestBodies = append(t.requestBodies, body)
		req.Body = io.NopCloser(bytes.NewReader(body))
	} else {
		t.requestBodies = append(t.requestBodies, nil)
	}
	t.requests = append(t.requests, req)
	if t.cursor >= len(t.responses) {
		return nil, fmt.Errorf("fakeTransport: out of canned responses (cursor=%d)", t.cursor)
	}
	r := t.responses[t.cursor]
	t.cursor++
	if r.Body != nil {
		r.Body = &countingCloser{ReadCloser: r.Body, parent: t}
	}
	return r, nil
}

type countingCloser struct {
	io.ReadCloser
	parent *fakeTransport
	closed bool
}

func (c *countingCloser) Close() error {
	if !c.closed {
		c.closed = true
		c.parent.closeCalls++
	}
	return c.ReadCloser.Close()
}

// okResponse builds a 200 application/json response from a body string.
func okResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

// sseResponse builds a 200 text/event-stream response from one or more SSE
// event blocks. Each event must include the trailing blank line.
func sseResponse(events ...string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(strings.Join(events, ""))),
	}
}

// newAgentWithTransport wires fakeTransport into an agent via WithClient.
func newAgentWithTransport(model string, ft *fakeTransport, opts ...Option) fugue.Agent {
	httpClient := &http.Client{Transport: ft}
	c := sdk.NewClient(option.WithHTTPClient(httpClient), option.WithAPIKey("test-key"))
	all := append([]Option{WithClient(&c)}, opts...)
	return New(model, all...)
}

// msg is a tiny helper for building text-only Messages in tests.
func msg(role fugue.Role, text string) fugue.Message {
	return fugue.Message{Role: role, Content: []fugue.Part{fugue.Text{Text: text}}}
}

func TestNew_PanicsOnEmptyModel(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("New(\"\") should panic")
		}
	}()
	New("")
}

func TestSplitSystem_FromOption(t *testing.T) {
	body := []fugue.Message{msg(fugue.RoleUser, "hello")}
	sys, rest, err := splitSystem(body, "you are helpful")
	if err != nil {
		t.Fatalf("splitSystem: %v", err)
	}
	if sys != "you are helpful" {
		t.Errorf("sys = %q, want %q", sys, "you are helpful")
	}
	if !reflect.DeepEqual(rest, body) {
		t.Errorf("rest = %v, want unchanged %v", rest, body)
	}
}

func TestSplitSystem_FromLeadingRoleSystem(t *testing.T) {
	in := []fugue.Message{
		msg(fugue.RoleSystem, "be brief"),
		msg(fugue.RoleSystem, "no markdown"),
		msg(fugue.RoleUser, "hello"),
	}
	sys, rest, err := splitSystem(in, "")
	if err != nil {
		t.Fatalf("splitSystem: %v", err)
	}
	wantSys := "be brief\nno markdown"
	if sys != wantSys {
		t.Errorf("sys = %q, want %q", sys, wantSys)
	}
	wantRest := []fugue.Message{msg(fugue.RoleUser, "hello")}
	if !reflect.DeepEqual(rest, wantRest) {
		t.Errorf("rest = %v, want %v", rest, wantRest)
	}
}

func TestSplitSystem_OptionOverridesLeadingRoleSystem(t *testing.T) {
	in := []fugue.Message{
		msg(fugue.RoleSystem, "be brief"),
		msg(fugue.RoleUser, "hello"),
	}
	sys, rest, err := splitSystem(in, "use option text")
	if err != nil {
		t.Fatalf("splitSystem: %v", err)
	}
	if sys != "use option text" {
		t.Errorf("sys = %q, want option text", sys)
	}
	wantRest := []fugue.Message{msg(fugue.RoleUser, "hello")}
	if !reflect.DeepEqual(rest, wantRest) {
		t.Errorf("rest = %v, want %v", rest, wantRest)
	}
}

func TestSplitSystem_MidConversationSystemErrors(t *testing.T) {
	in := []fugue.Message{
		msg(fugue.RoleUser, "hello"),
		msg(fugue.RoleSystem, "INTERRUPT"),
		msg(fugue.RoleAssistant, "hi"),
	}
	_, _, err := splitSystem(in, "")
	if err == nil {
		t.Fatal("expected error for mid-conversation RoleSystem")
	}
	if !strings.Contains(err.Error(), "system") {
		t.Errorf("error should mention system, got: %v", err)
	}
}

func TestToAPIMessages_EmptyInputErrors(t *testing.T) {
	if _, err := toAPIMessages(nil); err == nil {
		t.Fatal("expected error for nil input")
	}
	if _, err := toAPIMessages([]fugue.Message{}); err == nil {
		t.Fatal("expected error for zero-length input")
	}
}

// firstTextOf walks the first text block of a MessageParam.
func firstTextOf(m sdk.MessageParam) string {
	for _, b := range m.Content {
		if b.OfText != nil {
			return b.OfText.Text
		}
	}
	return ""
}

func TestToAPIMessages_TextRoundTrip(t *testing.T) {
	in := []fugue.Message{
		msg(fugue.RoleUser, "hello"),
		msg(fugue.RoleAssistant, "hi back"),
		msg(fugue.RoleUser, "how are you"),
	}
	got, err := toAPIMessages(in)
	if err != nil {
		t.Fatalf("toAPIMessages: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("want 3 messages, got %d", len(got))
	}
	wantRoles := []sdk.MessageParamRole{
		sdk.MessageParamRoleUser,
		sdk.MessageParamRoleAssistant,
		sdk.MessageParamRoleUser,
	}
	wantTexts := []string{"hello", "hi back", "how are you"}
	for i, m := range got {
		if m.Role != wantRoles[i] {
			t.Errorf("msg %d role = %q, want %q", i, m.Role, wantRoles[i])
		}
		if text := firstTextOf(m); text != wantTexts[i] {
			t.Errorf("msg %d text = %q, want %q", i, text, wantTexts[i])
		}
	}
}

func TestFromAPIResponse_TextOnly(t *testing.T) {
	// Build a Message manually. ContentBlockUnion is a flattened union with
	// Type discriminator + variant fields populated.
	resp := &sdk.Message{
		Content: []sdk.ContentBlockUnion{
			{Type: "text", Text: "hello world"},
		},
		StopReason: sdk.StopReasonEndTurn,
	}
	got, err := fromAPIResponse(resp)
	if err != nil {
		t.Fatalf("fromAPIResponse: %v", err)
	}
	if got.Role != fugue.RoleAssistant {
		t.Errorf("role = %v, want assistant", got.Role)
	}
	if len(got.Content) != 1 {
		t.Fatalf("want 1 content part, got %d", len(got.Content))
	}
	if txt, ok := got.Content[0].(fugue.Text); !ok || txt.Text != "hello world" {
		t.Errorf("content[0] = %v, want Text{hello world}", got.Content[0])
	}
	if got.Name != string(sdk.StopReasonEndTurn) {
		t.Errorf("Name = %q, want %q", got.Name, sdk.StopReasonEndTurn)
	}
}

// hasBase64ImageBlock returns true if m has an image block whose source is
// base64 with the given media type and decoded bytes equal to wantBytes.
func hasBase64ImageBlock(m sdk.MessageParam, mime string, wantBytes []byte) bool {
	for _, b := range m.Content {
		if b.OfImage == nil {
			continue
		}
		src := b.OfImage.Source.OfBase64
		if src == nil {
			continue
		}
		if string(src.MediaType) == mime && src.Data == base64.StdEncoding.EncodeToString(wantBytes) {
			return true
		}
	}
	return false
}

// hasURLImageBlock returns true if m has an image block with source URL == wantURL.
func hasURLImageBlock(m sdk.MessageParam, wantURL string) bool {
	for _, b := range m.Content {
		if b.OfImage == nil {
			continue
		}
		src := b.OfImage.Source.OfURL
		if src == nil {
			continue
		}
		if src.URL == wantURL {
			return true
		}
	}
	return false
}

func TestToAPIMessages_ImageWithData(t *testing.T) {
	imgBytes := []byte{0x89, 0x50, 0x4E, 0x47}
	in := []fugue.Message{{
		Role: fugue.RoleUser,
		Content: []fugue.Part{
			fugue.Image{MIMEType: "image/png", Data: imgBytes},
			fugue.Text{Text: "what's in this image?"},
		},
	}}
	got, err := toAPIMessages(in)
	if err != nil {
		t.Fatalf("toAPIMessages: %v", err)
	}
	if !hasBase64ImageBlock(got[0], "image/png", imgBytes) {
		t.Errorf("expected base64 image block in message content")
	}
}

func TestToAPIMessages_ImageWithURL(t *testing.T) {
	in := []fugue.Message{{
		Role: fugue.RoleUser,
		Content: []fugue.Part{
			fugue.Image{URL: "https://example.com/x.png"},
		},
	}}
	got, err := toAPIMessages(in)
	if err != nil {
		t.Fatalf("toAPIMessages: %v", err)
	}
	if !hasURLImageBlock(got[0], "https://example.com/x.png") {
		t.Errorf("expected URL image block")
	}
}
